import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
import torch
import json
import open3d as o3d
from collections import defaultdict

from .factory import register_dataset
from .base_dataset import BaseDataset
from utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg, sample_sparse_depth
from dvgt.utils.geometry import closed_form_inverse_se3
from utils.transformer_np import project_lidar_to_depth
from utils.tools import gen_and_create_output_dirs
from utils.io import read_image

logger = logging.getLogger(__name__)

@register_dataset('waymo')
class WaymoDataset(BaseDataset):

    CAMERA_NAMES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    
    VAL_SCENE_NAMES = {
        '10335539493577748957_1372_870_1392_870', '12102100359426069856_3931_470_3951_470', '9114112687541091312_1100_000_1120_000', '10247954040621004675_2180_000_2200_000', 
        '14931160836268555821_5778_870_5798_870', '260994483494315994_2797_545_2817_545', '272435602399417322_2884_130_2904_130', '5832416115092350434_60_000_80_000', 
        '12358364923781697038_2232_990_2252_990', '6491418762940479413_6520_000_6540_000', '967082162553397800_5102_900_5122_900', '9265793588137545201_2981_960_3001_960', 
        '2736377008667623133_2676_410_2696_410', '14811410906788672189_373_113_393_113', '10289507859301986274_4200_000_4220_000', '8888517708810165484_1549_770_1569_770', 
        '8956556778987472864_3404_790_3424_790', '4490196167747784364_616_569_636_569', '18331704533904883545_1560_000_1580_000', '15028688279822984888_1560_000_1580_000', 
        '4409585400955983988_3500_470_3520_470', '9231652062943496183_1740_000_1760_000', '8506432817378693815_4860_000_4880_000', '11356601648124485814_409_000_429_000', 
        '6001094526418694294_4609_470_4629_470', '16767575238225610271_5185_000_5205_000', '18252111882875503115_378_471_398_471', '89454214745557131_3160_000_3180_000', 
        '13982731384839979987_1680_000_1700_000', 'LICENSE', '10359308928573410754_720_000_740_000', '14486517341017504003_3406_349_3426_349', '13184115878756336167_1354_000_1374_000', 
        '1105338229944737854_1280_000_1300_000', '12496433400137459534_120_000_140_000', '18045724074935084846_6615_900_6635_900', '14663356589561275673_935_195_955_195', 
        '12134738431513647889_3118_000_3138_000', '9579041874842301407_1300_000_1320_000', '7799643635310185714_680_000_700_000', '902001779062034993_2880_000_2900_000', 
        '11434627589960744626_4829_660_4849_660', '14739149465358076158_4740_000_4760_000', '16213317953898915772_1597_170_1617_170', '7650923902987369309_2380_000_2400_000', 
        '10689101165701914459_2072_300_2092_300', '7932945205197754811_780_000_800_000', '14300007604205869133_1160_000_1180_000', '3126522626440597519_806_440_826_440', 
        '8845277173853189216_3828_530_3848_530', '1906113358876584689_1359_560_1379_560', '10837554759555844344_6525_000_6545_000', '16229547658178627464_380_000_400_000', 
        '15488266120477489949_3162_920_3182_920', '5847910688643719375_180_000_200_000', '30779396576054160_1880_000_1900_000', '15611747084548773814_3740_000_3760_000', 
        '4764167778917495793_860_000_880_000', '5372281728627437618_2005_000_2025_000', '16751706457322889693_4475_240_4495_240', '12866817684252793621_480_000_500_000', 
        '3915587593663172342_10_000_30_000', '4759225533437988401_800_000_820_000', '2094681306939952000_2972_300_2992_300', '13694146168933185611_800_000_820_000', 
        '12940710315541930162_2660_000_2680_000', '12306251798468767010_560_000_580_000', '7253952751374634065_1100_000_1120_000', '15724298772299989727_5386_410_5406_410', 
        '8079607115087394458_1240_000_1260_000', '13336883034283882790_7100_000_7120_000', '5574146396199253121_6759_360_6779_360', '1505698981571943321_1186_773_1206_773', 
        '9164052963393400298_4692_970_4712_970', '7119831293178745002_1094_720_1114_720', '346889320598157350_798_187_818_187', '1331771191699435763_440_000_460_000', 
        '6324079979569135086_2372_300_2392_300', '17539775446039009812_440_000_460_000', '1405149198253600237_160_000_180_000', '13469905891836363794_4429_660_4449_660', 
        '8133434654699693993_1162_020_1182_020', '9472420603764812147_850_000_870_000', '17860546506509760757_6040_000_6060_000', '5302885587058866068_320_000_340_000', 
        '3015436519694987712_1300_000_1320_000', '8302000153252334863_6020_000_6040_000', '17626999143001784258_2760_000_2780_000', '2506799708748258165_6455_000_6475_000', 
        '4013125682946523088_3540_000_3560_000', '8331804655557290264_4351_740_4371_740', '3039251927598134881_1240_610_1260_610', '3077229433993844199_1080_000_1100_000', 
        '17344036177686610008_7852_160_7872_160', '2834723872140855871_1615_000_1635_000', '1464917900451858484_1960_000_1980_000', '933621182106051783_4160_000_4180_000', 
        '17065833287841703_2980_000_3000_000', '14127943473592757944_2068_000_2088_000', '12657584952502228282_3940_000_3960_000', '15948509588157321530_7187_290_7207_290', 
        '10868756386479184868_3000_000_3020_000', '17612470202990834368_2800_000_2820_000', '14956919859981065721_1759_980_1779_980', '18024188333634186656_1566_600_1586_600', 
        '4816728784073043251_5273_410_5293_410', '191862526745161106_1400_000_1420_000', '10448102132863604198_472_000_492_000', '11660186733224028707_420_000_440_000', 
        '2624187140172428292_73_000_93_000', '10203656353524179475_7625_000_7645_000', '662188686397364823_3248_800_3268_800', '9024872035982010942_2578_810_2598_810', 
        '11450298750351730790_1431_750_1451_750', '4195774665746097799_7300_960_7320_960', '11048712972908676520_545_000_565_000', '6637600600814023975_2235_000_2255_000', 
        '3651243243762122041_3920_000_3940_000', '13299463771883949918_4240_000_4260_000', '4854173791890687260_2880_000_2900_000', '17152649515605309595_3440_000_3460_000', 
        '6680764940003341232_2260_000_2280_000', '4690718861228194910_1980_000_2000_000', '14107757919671295130_3546_370_3566_370', '12820461091157089924_5202_916_5222_916', 
        '11037651371539287009_77_670_97_670', '18333922070582247333_320_280_340_280', '15021599536622641101_556_150_576_150', '5373876050695013404_3817_170_3837_170', 
        '4612525129938501780_340_000_360_000', '15496233046893489569_4551_550_4571_550', '17136314889476348164_979_560_999_560', '6074871217133456543_1000_000_1020_000', 
        '18305329035161925340_4466_730_4486_730', '13941626351027979229_3363_930_3383_930', '11387395026864348975_3820_000_3840_000', '5183174891274719570_3464_030_3484_030', 
        '8137195482049459160_3100_000_3120_000', '15959580576639476066_5087_580_5107_580', '17244566492658384963_2540_000_2560_000', '2105808889850693535_2295_720_2315_720', 
        '6183008573786657189_5414_000_5434_000', '14333744981238305769_5658_260_5678_260', '7732779227944176527_2120_000_2140_000', '16979882728032305374_2719_000_2739_000', 
        '7988627150403732100_1487_540_1507_540', '14687328292438466674_892_000_912_000', '2367305900055174138_1881_827_1901_827', '1943605865180232897_680_000_700_000', 
        '366934253670232570_2229_530_2249_530', '12831741023324393102_2673_230_2693_230', '8907419590259234067_1960_000_1980_000', '6161542573106757148_585_030_605_030', 
        '17962792089966876718_2210_933_2230_933', '2308204418431899833_3575_000_3595_000', '9443948810903981522_6538_870_6558_870', '271338158136329280_2541_070_2561_070', 
        '5772016415301528777_1400_000_1420_000', '15224741240438106736_960_000_980_000', '1071392229495085036_1844_790_1864_790', '3577352947946244999_3980_000_4000_000', 
        '15096340672898807711_3765_000_3785_000', '16204463896543764114_5340_000_5360_000', '15396462829361334065_4265_000_4285_000', '18446264979321894359_3700_000_3720_000', 
        '11406166561185637285_1753_750_1773_750', '14244512075981557183_1226_840_1246_840', '17135518413411879545_1480_000_1500_000', '2335854536382166371_2709_426_2729_426', 
        '17791493328130181905_1480_000_1500_000', '3731719923709458059_1540_000_1560_000', '5289247502039512990_2640_000_2660_000', '12374656037744638388_1412_711_1432_711', 
        '4426410228514970291_1620_000_1640_000', '9243656068381062947_1297_428_1317_428', '8679184381783013073_7740_000_7760_000', '17703234244970638241_220_000_240_000', 
        '4423389401016162461_4235_900_4255_900', '14383152291533557785_240_000_260_000', '14165166478774180053_1786_000_1806_000', '8398516118967750070_3958_000_3978_000', 
        '17694030326265859208_2340_000_2360_000', '11901761444769610243_556_000_576_000', '13178092897340078601_5118_604_5138_604', '13573359675885893802_1985_970_2005_970', 
        '14624061243736004421_1840_000_1860_000', '7493781117404461396_2140_000_2160_000', '1024360143612057520_3580_000_3600_000', '1457696187335927618_595_027_615_027', 
        '17763730878219536361_3144_635_3164_635', '14081240615915270380_4399_000_4419_000', '13356997604177841771_3360_000_3380_000', '14262448332225315249_1280_000_1300_000', 
        '2551868399007287341_3100_000_3120_000', '11616035176233595745_3548_820_3568_820', '447576862407975570_4360_000_4380_000', '6707256092020422936_2352_392_2372_392', 
        '5990032395956045002_6600_000_6620_000', '4246537812751004276_1560_000_1580_000', '9041488218266405018_6454_030_6474_030', '4575389405178805994_4900_000_4920_000',
        '13415985003725220451_6163_000_6183_000', '7163140554846378423_2717_820_2737_820', 
    }

    def __init__(
        self,
        data_root: str = 'public_datasets/waymo_ns',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/waymo',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/waymo',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/waymo',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/waymo',
        num_tokens: int = 3700,
        split: str = 'train',    # 'train' or 'val'
    ) -> None:
        self.split = split
        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)
        self.T_rub_rdf = np.array([
            [ 1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0, -1, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float64)

    @dataclass(frozen=True, slots=True)
    class SampleData:
        scene_id: str
        frame_idx: int
        cam_type: str
        filename: Path
        frame_meta: Dict
        lidar_meta: Dict
        sensor_meta: Dict

    def _gather_samples(self) -> List[SampleData]:
        if self.split == 'train':
            scene_dirs = [d for d in self.data_root.iterdir() if d.name not in self.VAL_SCENE_NAMES]
        else:
            scene_dirs = [d for d in self.data_root.iterdir() if d.name in self.VAL_SCENE_NAMES]

        samples_to_process = []
        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            transforms_path = scene_dir / "transforms.json"

            with open(transforms_path, 'r') as f:
                clip_meta = json.load(f)

            # 按照CAM TYPE对meta frame分组
            cam_info_dict = defaultdict(list)
            for frame_meta in clip_meta['frames']:
                cam_info_dict[frame_meta['camera']].append(frame_meta)

            for cam_type, frame_infos in cam_info_dict.items():
                for frame_idx, frame_info in enumerate(frame_infos):
                    filename = (Path(scene_id) / frame_info['file_path']).with_suffix('')
                    assert frame_info['timestamp'] == clip_meta['lidar_frames'][frame_idx]['timestamp']
                    samples_to_process.append(self.SampleData(
                        scene_id=scene_id,
                        frame_idx=frame_idx,
                        cam_type=cam_type,
                        filename=filename,
                        frame_meta=frame_info,
                        lidar_meta=clip_meta['lidar_frames'][frame_idx],
                        sensor_meta=clip_meta['sensor_params'][cam_type]
                    ))
        
        logging.info(f"处理：{len(samples_to_process)}条数据")
        return samples_to_process

    @staticmethod
    def _convert_opengl_to_opencv(se3: np.ndarray) -> np.ndarray:
        return np.concatenate([ se3[:,0:1], 
                               -se3[:,1:2], 
                               -se3[:,2:3], 
                                se3[:,3:4]], axis=-1)

    def _get_image_and_meta_info(self, item: SampleData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_path = self.data_root / item.scene_id / item.frame_meta['file_path']
        image = read_image(image_path)
        original_aspect_ratio = image.shape[1] / image.shape[0]

        # Intrinsics and Distortion
        cam_intrinsics = np.array(item.sensor_meta['camera_intrinsic'])
        distortion = np.array(item.sensor_meta['camera_D'])
        
        # Undistort image and normalize intrinsics
        image_input, cam_intrinsics_input, _ = process_image_sequentially(image, cam_intrinsics, distortion, num_tokens=self.num_tokens)
        
        # Resize image while maintaining aspect ratio
        image_save, cam_intrinsics_save = resize_to_aspect_ratio(image_input, cam_intrinsics_input, original_aspect_ratio)

        # c2w
        T_world_cam_rub = np.array(item.frame_meta['transform_matrix'])
        T_world_cam_rdf = T_world_cam_rub @ self.T_rub_rdf
        T_cam_rdf_world = closed_form_inverse_se3(T_world_cam_rdf)

        # lidar proj depth需要
        T_ego_flu_cam_rub = np.array(item.sensor_meta['extrinsic'])
        T_ego_flu_cam_rdf = T_ego_flu_cam_rub @ self.T_rub_rdf
        T_cam_rdf_ego_flu = closed_form_inverse_se3(T_ego_flu_cam_rdf)
        
        # ego 2 world == lidar 2 world in waymo
        T_world_ego_flu = np.array(item.lidar_meta['transform_matrix'])
        T_world_ego_rdf = T_world_ego_flu @ self.T_flu_rdf
        
        return (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, T_cam_rdf_world[:3], T_world_ego_rdf[:3],
            T_cam_rdf_ego_flu, 
        )

    def _get_proj_depth(self, item: SampleData, cam_intrinsics: np.ndarray, T_cam_rdf_ego_flu: np.ndarray, height: int, width: int) -> torch.Tensor:
        lidar_path = self.data_root / item.scene_id / item.lidar_meta['file_path']
        pcd = o3d.io.read_point_cloud(str(lidar_path))
        lidar_points = np.asarray(pcd.points)  # (N, 3), in vehicle frame

        # T_cam_rdf_ego_flu 等价于 T_cam_rdf_lidar
        proj_depth = project_lidar_to_depth(
            points=lidar_points,
            points_coordinate_to_camera=T_cam_rdf_ego_flu,
            camera_intrinsic=cam_intrinsics,
            height=height, 
            width=width
        )
        return proj_depth

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        data_dict = {}
        # 生成保存路径
        data_dict.update(
            gen_and_create_output_dirs(
                item.filename, self.storage_pred_depth_path, self.storage_proj_depth_path, 
                self.storage_align_depth_path, self.storage_image_path
            )
        )

        # meta info
        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, vehicle_to_world,
            T_cam_rdf_ego_flu, 
        )= self._get_image_and_meta_info(item)

        data_dict.update({
            'scene_id': item.scene_id,
            'frame_idx': item.frame_idx,
            'cam_type': item.cam_type,
            'filename': str(item.filename),
            'intrinsics': cam_intrinsics_save.flatten().tolist(),
            'extrinsics': cam_extrinsics.flatten().tolist(),
            'ego_to_world': vehicle_to_world.flatten().tolist(),
        })

        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]

        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])
        
        # MoGe input
        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,    # float
        })
        
        proj_depth = self._get_proj_depth(item, cam_intrinsics_save, T_cam_rdf_ego_flu, H_save, W_save)

        data_dict.update({
            'proj_depth': torch.from_numpy(proj_depth).to(torch.float32),
            'image_save': image_save,     # 后续可以直接保存这个 uint8 的图片
            'image_size': (H_input, W_input)
        })

        return data_dict