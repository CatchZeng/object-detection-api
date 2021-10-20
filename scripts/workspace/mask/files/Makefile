gen-tfrecord:
	python labelme2coco.py images/train --output images/train.json
	python labelme2coco.py images/val --output images/val.json

	python create_coco_tf_record.py --logtostderr \
	--train_image_dir=images/train \
	--val_image_dir=images/val \
	--train_annotations_file=images/train.json \
	--val_annotations_file=images/val.json \
	--include_masks=True \
	--output_dir=./annotations

	python gen_label_map.py
dl-model:
	mkdir -p pre-trained-models; \
	model=mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8; \
	curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
train:
	python model_main_tf2.py \
	--model_dir=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 \
	--pipeline_config_path=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config
eval:
	python model_main_tf2.py \
    --pipeline_config_path=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config \
    --model_dir=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 \
    --checkpoint_dir=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 \
    --alsologtostderr
.PHONY: export
export:
	python exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config \
	--trained_checkpoint_dir models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/ \
	--output_directory exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
serving:
	docker pull tensorflow/serving
	docker run -d --name serving_base tensorflow/serving
	# https://stackoverflow.com/a/45552938/14785930
	docker exec -it serving_base mkdir -p /models/{{.Name}}/1
	docker cp exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/saved_model/. serving_base:/models/{{.Name}}/1
	docker commit --change "ENV MODEL_NAME {{.Name}}" serving_base {{.Name}}
	docker rm -f serving_base
serving-test:
	python deployments/serving/client.py
flask:
	cp annotations/label_map.pbtxt deployments/flask
	cd deployments/flask && make build