"""
Author: https://github.com/LightwheelAI/street-gaussians-ns/blob/main/scripts/pythons/extract_waymo.py

The results follow https://docs.nerf.studio/quickstart/data_conventions.html
transform_matrix is the camera-to-world matrix
"""
import argparse
import json
import numpy as np
import tensorflow as tf
try:
    import open3d as o3d
except:
    import open3d_pycg as o3d

from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any, Union
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
from google.protobuf import json_format
from tqdm import tqdm

DEBUG = False

if int(tf.__version__.split(".")[0]) < 2:
    tf.enable_eager_execution()


class WaymoDataExtractor:
    RETURN_OK = 0
    RETURN_SKIP = 1

    MIN_MOVING_SPEED = 0.2

    def __init__(self, waymo_root: Union[Path, str], num_workers: int, no_lidar: bool, no_camera: bool) -> None:
        self.waymo_root = Path(waymo_root)
        self.num_workers = num_workers
        self.no_lidar = no_lidar
        self.no_camera = no_camera

        self._box_type_to_str = {
            label_pb2.Label.Type.TYPE_UNKNOWN: "unknown",
            label_pb2.Label.Type.TYPE_VEHICLE: "car",
            label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
            label_pb2.Label.Type.TYPE_SIGN: "sign",
            label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
        }

    def extract_all(self, split, specify_segments: List[str], out_root: Union[Path, str]):
        # if out_root not exists, create it
        out_root = Path(out_root)
        if not out_root.exists():
            out_root.mkdir(parents=True)

        all_segments = self.list_segments(split)

        def find_segement(partial_segment_name: str, segments: List[Path]):
            for seg in segments:
                if partial_segment_name in seg.as_posix():
                    return seg
            return None

        inexist_segs, task_segs = [], []
        if specify_segments:
            for specify_segment in specify_segments:
                seg = find_segement(specify_segment, all_segments)
                if seg is None:
                    inexist_segs.append(specify_segment)
                else:
                    task_segs.append(seg)
        else:
            task_segs = all_segments

        if inexist_segs:
            print(f"{len(inexist_segs)} segments not found:")
            for seg in inexist_segs:
                print(seg)

        def print_error(e):
            print("ERROR:", e)

        fail_tasks, skip_tasks, succ_tasks = [], [], []
        
        if DEBUG:
            self.extract_one(task_segs[0], out_root if split is None else out_root / split)

        with Pool(processes=self.num_workers) as pool:
            results = [
                # pool.apply_async(func=self.extract_one, args=(seg, out_root if split is None else out_root / split), error_callback=print_error)
                pool.apply_async(func=self.extract_one, args=(seg, out_root), error_callback=print_error)
                for seg in task_segs
            ]

            for result in results:
                result.wait()

            for segment, result in zip(task_segs, results):
                if not result.successful():
                    fail_tasks.append(segment)
                elif result.get() == WaymoDataExtractor.RETURN_SKIP:
                    skip_tasks.append(segment)
                elif result.get() == WaymoDataExtractor.RETURN_OK:
                    succ_tasks.append(segment)

        print(
            f"""{len(task_segs)} tasks total, {len(fail_tasks)} tasks failed, """
            f"""{len(skip_tasks)} tasks skipped, {len(succ_tasks)} tasks success"""
        )
        print("Failed tasks:")
        for seg in fail_tasks:
            print(seg.as_posix())
        print("Skipped tasks:")
        for seg in skip_tasks:
            print(seg.as_posix())

    def list_segments(self, split=None) -> List[str]:
        if split is None:
            return list(self.waymo_root.glob("*.tfrecord"))
        else:
            return list((self.waymo_root / split).glob("*.tfrecord"))

    def extract_one(self, segment_tfrecord: Path, out_dir: Path) -> int:
        dataset = tf.data.TFRecordDataset(segment_tfrecord.as_posix(), compression_type="")
        segment_name = None
        segment_out_dir = None
        sensor_params = None
        camera_frames = []
        lidar_frames = []
        annotations = []

        for frame_idx, data in tqdm(enumerate(dataset)):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            segment_name = frame.context.name
            if frame_idx == 0:
                segment_out_dir = out_dir / segment_name
                if (segment_out_dir / "transforms.json").exists() and \
                   (segment_out_dir / "annotation.json").exists() and \
                   (segment_out_dir / "maps.json").exists():
                    return WaymoDataExtractor.RETURN_SKIP

                sensor_params = self.extract_sensor_params(frame)
                map_data = self.extract_map_data(frame)
                # record the vehicle pose at the first frame
                vehicle_init_pose = np.array(frame.pose.transform).reshape((4, 4))

            camera_frames.extend(self.extact_frame_images(frame, segment_out_dir, sensor_params))
            lidar_frames.extend(self.extract_frame_lidars(frame, segment_out_dir, sensor_params))
            annotations.append(self.extract_frame_annotation(frame))

        camera_frames.sort(key=lambda frame: f"{frame['file_path']}")
        lidar_frames.sort(key=lambda frame: f"{frame['file_path']}")
        meta = {"sensor_params": sensor_params, "frames": camera_frames, "lidar_frames": lidar_frames, "vehicle_init_pose": vehicle_init_pose.tolist()}

        with open(segment_out_dir / "transforms.json", "w") as fout:
            json.dump(meta, fout, indent=4)
        with open(segment_out_dir / "annotation.json", "w") as fout:
            json.dump({"frames": annotations}, fout, indent=4)
        with open(segment_out_dir / "maps.json", "w") as fout:
            json.dump(map_data, fout, indent=4)

        return WaymoDataExtractor.RETURN_OK

    def extract_sensor_params(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        out = {"camera_order": [self.get_camera_name(i) for i in [1, 2, 4, 5, 3]]} # counterclockwise from front
        for camera_calib in frame.context.camera_calibrations:
            camera_name = self.get_camera_name(camera_calib.name)

            intrinsic = camera_calib.intrinsic
            fx, fy, cx, cy = intrinsic[:4]
            distortion = intrinsic[4:]

            extrinsic = np.array(camera_calib.extrinsic.transform).reshape((4, 4)) # camera-to-vehicle matrix. camera is FLU order!
            # Convert to nerfstudio/blender camera coord. FLU -> RUB
            new_X = - extrinsic[:, 1:2]
            new_Y = extrinsic[:, 2:3]
            new_Z = - extrinsic[:, 0:1]
            extrinsic = np.concatenate([new_X, new_Y, new_Z, extrinsic[:, 3:4]], axis=1)

            out[camera_name] = {
                "type": "camera",
                "camera_model": "OPENCV",
                "camera_intrinsic": [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                "camera_D": distortion,
                "extrinsic": extrinsic.tolist(), # Trasformation from camera frame to vehicle frame.
                "width": camera_calib.width,
                "height": camera_calib.height,
            }

        for lidar_calib in frame.context.laser_calibrations:
            lidar_name = self.get_lidar_name(lidar_calib.name)
            extrinsic = np.array(lidar_calib.extrinsic.transform).reshape((4, 4)) # Trasformation from LiDAR frame to vehicle frame.
            out[lidar_name] = {"type": "lidar", "extrinsic": extrinsic.tolist()}

        return out

    def extract_map_data(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        """
        frame.map_features have many MapFeature item
        message MapFeature {
            // A unique ID to identify this feature.
            optional int64 id = 1;

            // Type specific data.
            oneof feature_data {
                LaneCenter lane = 3; # polyline
                RoadLine road_line = 4; # polyline
                RoadEdge road_edge = 5; # polyline
                StopSign stop_sign = 7; 
                Crosswalk crosswalk = 8; # polygon
                SpeedBump speed_bump = 9; # polygon
                Driveway driveway = 10; # polygon
            }
        }

        Returns:
            map_data: Dict
                'lane': list of polylines, each polyline is noted by several vertices.
                'road_line': list of polylines, each polyline is noted by several vertices.
                ...
        """
        def hump_to_underline(hump_str):
            import re
            return re.sub(r'([a-z])([A-Z])', r'\1_\2', hump_str).lower()

        map_features_list = json_format.MessageToDict(frame)['mapFeatures']
        feature_names = ["lane", "road_line", "road_edge", "crosswalk", "speed_bump", "driveway"]
        map_data = dict(zip(feature_names, [[] for _ in range(len(feature_names))]))

        for feature in tqdm(map_features_list):
            feature_name = list(feature.keys())
            feature_name.remove("id")
            feature_name = feature_name[0]
            feature_name_lower = hump_to_underline(feature_name)

            feature_content = feature[feature_name]
            if feature_name_lower in ["lane", "road_line", "road_edge"]:
                polyline = feature_content['polyline'] # [{'x':..., 'y':..., 'z':...}, {'x':..., 'y':..., 'z':...}, ...]
            elif feature_name_lower in ["crosswalk", "speed_bump", "driveway"]:
                polyline = feature_content['polygon'] # [{'x':..., 'y':..., 'z':...}, {'x':..., 'y':..., 'z':...}, ...]
            else:
                continue

            polyline = [[point['x'], point['y'], point['z']] for point in polyline] # [[x, y, z], [x, y, z], ...]
            map_data[hump_to_underline(feature_name)].append(polyline)

        return map_data

    def extact_frame_images(
        self, frame: dataset_pb2.Frame, segment_out_dir: Path, sensor_params
    ) -> List[Dict[str, Any]]:
        lidar_timestamp: int = frame.timestamp_micros
        frame_images = []

        for image_data in frame.images:
            camera_name = self.get_camera_name(image_data.name)

            save_path = segment_out_dir / "images" / camera_name / f"{lidar_timestamp}.jpg"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)

            if not self.no_camera:
                with open(save_path, "wb") as fp:
                    fp.write(image_data.image)
                
            # 4x4 float32 array with the vehicle pose at the timestamp of this camera image.
            # vehicle pose is vehicle-to-world matrix
            ego_pose = np.array(image_data.pose.transform).reshape((4, 4)) 
            camera_params = sensor_params[camera_name]
            intrinsic = np.array(camera_params["camera_intrinsic"])
            extrinsic = np.array(camera_params["extrinsic"])  # camera-to-vehicle, not that extrinsic contains FLU -> RUB transformation
            distortion = camera_params["camera_D"]
            camera2world = ego_pose @ extrinsic  


            frame_images.append(
                {
                    "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                    "fl_x": intrinsic[0, 0],
                    "fl_y": intrinsic[1, 1],
                    "cx": intrinsic[0, 2],
                    "cy": intrinsic[1, 2],
                    "w": camera_params["width"],
                    "h": camera_params["height"],
                    "camera_model": "OPENCV",
                    "camera": camera_name,
                    "timestamp": lidar_timestamp / 1.0e6,
                    "k1": distortion[0],
                    "k2": distortion[1],
                    "k3": distortion[4],
                    "k4": 0.0,
                    "p1": distortion[2],
                    "p2": distortion[3],
                    "transform_matrix": camera2world.tolist(),
                }
            )

        return frame_images

    def extract_frame_lidars(
        self, frame: dataset_pb2.Frame, segment_out_dir: Path, sensor_params
    ) -> List[Dict[str, Any]]:
        lidar_timestamp: int = frame.timestamp_micros
        frame_lidars = []
        pose = np.array(frame.pose.transform).reshape((4, 4))

        if self.no_lidar:
            points = [None] * len(frame.context.laser_calibrations)
        else:
            (
                range_images,
                camera_projections,
                _,
                range_image_top_pose,
            ) = frame_utils.parse_range_image_and_camera_projection(frame)
            # note pose = np.array(frame.pose.transform).reshape((4, 4))
            # and "transform_matrix": pose.tolist(),
            # so points are in vehicle' frame!! It has nothing to do with lidar extrinsics
            # because frame_utils already handle it!
            points, _ = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose
            )
            points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=1
            )
            # 3d points in vehicle frame. Confirmed.
            points = [np.concatenate([p1, p2]) for p1, p2 in zip(points, points_ri2)]
        lidar_ids = [calib.name for calib in frame.context.laser_calibrations]
        lidar_ids.sort()
        
        for lidar_id, lidar_points in zip(lidar_ids, points):
            lidar_name = self.get_lidar_name(lidar_id)
            
            save_path = segment_out_dir / "lidars" / lidar_name / f"{lidar_timestamp}.pcd"

            if lidar_name != "lidar_TOP":
                continue
            
            # save lidar points
            if not self.no_lidar:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(lidar_points)
                if not save_path.parent.exists():
                    save_path.parent.mkdir(parents=True)
                o3d.io.write_point_cloud(save_path.as_posix(), pcd)

            frame_lidars.append(
                {
                    "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                    "lidar": lidar_name,
                    "timestamp": lidar_timestamp / 1.0e6,
                    "transform_matrix": pose.tolist(),
                }
            )

        return frame_lidars

    def extract_frame_annotation(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        pose = np.array(frame.pose.transform).reshape((4, 4))
        objects = []
        for label in frame.laser_labels:
            center_vcs = np.array([label.box.center_x, label.box.center_y, label.box.center_z, 1])
            center_wcs = pose @ center_vcs
            heading = label.box.heading
            rotation_vcs = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
            rotation_wcs = pose[:3, :3] @ rotation_vcs
            rotation_wcs = R.from_matrix(rotation_wcs).as_quat()

            speed = np.sqrt(label.metadata.speed_x**2 + label.metadata.speed_y**2 + label.metadata.speed_z**2)

            objects.append(
                {
                    "type": self._box_type_to_str[label.type],
                    "gid": label.id,
                    "translation": center_wcs[:3].tolist(),
                    "size": [label.box.length, label.box.width, label.box.height],
                    "rotation": [rotation_wcs[3], rotation_wcs[0], rotation_wcs[1], rotation_wcs[2]],
                    "is_moving": bool(speed > self.MIN_MOVING_SPEED)
                }
            )

        return {"timestamp": frame.timestamp_micros / 1.0e6, "objects": objects}

    def get_camera_name(self, name_int) -> str:
        return dataset_pb2.CameraName.Name.Name(name_int)

    def get_lidar_name(self, name_int) -> str:
        # Avoid using same names with cameras
        return "lidar_" + dataset_pb2.LaserName.Name.Name(name_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Root directory of waymo dataset (tfrecord files).")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory of extracted data.")
    parser.add_argument("--no_lidar", action="store_true", help="Do not extract lidar data.")
    parser.add_argument("--no_camera", action="store_true", help="Do not extract camera data.")
    parser.add_argument("--split", "-s", default=None, const=None, nargs="?", choices=["training", "testing", "validation", "train", "vali", None], 
                        help="Split of the dataset. \
                              If 'training' or 'testing' is specified, suppose you have a 2-level heirarchy of input_dir. \
                              If None is specified, suppose you have a 1-level heirarchy of input_dir."
                        )
    parser.add_argument("--specify_segments", default=[], nargs="+")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    extractor = WaymoDataExtractor(args.input_dir, args.num_workers, args.no_lidar, args.no_camera)
    extractor.extract_all(args.split, args.specify_segments, args.output_dir)
