# copy from https://github.com/OpenDriveLab/OpenScene.git

# Scripts for processing nuplan data.

# Please install https://github.com/autonomousvision/navsim.
# This is used for generating High-Level Driving Commands, used in E2E AD.
export NAVSIM_DEVKIT_ROOT=third_party/navsim_v1_1
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}

split=mini

# Please download all the nuplan data from https://www.nuscenes.org/nuplan.
export NUPLAN_PATH=public_datasets/nuplan-v1.1
export NUPLAN_DB_PATH=${NUPLAN_PATH}/splits/${split}
export NUPLAN_SENSOR_PATH=${NUPLAN_PATH}/sensor_blobs
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=${NUPLAN_PATH}/maps

OUT_DIR=public_datasets/nuplan_full/meta/${split}

python create_openscene_metadata.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version ${NUPLAN_MAP_VERSION} \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 128 \
  --sample-interval 2   # nuplan是20hz，我们需要标注10hz的
