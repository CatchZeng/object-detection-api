# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash

.PHONY: models
models:
	if [ -d "./models" ]; then \
	    echo 'models downloaded'; \
	  else \
        git clone --depth=1 https://github.com/tensorflow/models; \
	fi
api:
	cd models/research && \
	protoc object_detection/protos/*.proto --python_out=. && \
	cp object_detection/packages/tf2/setup.py . && \
	python -m pip install --use-feature=2020-resolver .
test:
	cd models/research && \
	python object_detection/builders/model_builder_tf2_test.py
install:  models  api test
workspace-box:
	python scripts/workspace/box/workspace.py --save_dir=$(SAVE_DIR) --name=$(NAME)
	cp -r scripts/workspace/box/files/* $(SAVE_DIR)/$(NAME)
	cp models/research/object_detection/model_main_tf2.py $(SAVE_DIR)/$(NAME)
	cp models/research/object_detection/exporter_main_v2.py $(SAVE_DIR)/$(NAME)
	cp models/research/object_detection/export_tflite_graph_tf2.py $(SAVE_DIR)/$(NAME)
	touch $(SAVE_DIR)/$(NAME)/annotations/label_map.pbtxt
workspace-mask:
	python scripts/workspace/mask/workspace.py --save_dir=$(SAVE_DIR) --name=$(NAME)
	cp -r scripts/workspace/mask/files/* $(SAVE_DIR)/$(NAME)
	python scripts/workspace/mask/replace.py --dir=$(SAVE_DIR)/$(NAME) --pattern="{{.Name}}" --repl=$(NAME)
	cp models/research/object_detection/model_main_tf2.py $(SAVE_DIR)/$(NAME)
	cp models/research/object_detection/exporter_main_v2.py $(SAVE_DIR)/$(NAME)
	touch $(SAVE_DIR)/$(NAME)/annotations/label_map.pbtxt