# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: models
models:
	if [ -d "./models" ]; then \
	    echo 'models downloaded'; \
	  else \
        git clone --depth=1 https://github.com/tensorflow/models; \
	fi
conda:
	conda create -n od python=3.8.5
	$(CONDA_ACTIVATE) od
api:
	$(CONDA_ACTIVATE) od
	cd models/research && \
	protoc object_detection/protos/*.proto --python_out=. && \
	cp object_detection/packages/tf2/setup.py . && \
	python -m pip install .
test:
	$(CONDA_ACTIVATE) od
	cd models/research && \
	python object_detection/builders/model_builder_tf2_test.py
install: conda models api test
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
	cp models/research/object_detection/model_main_tf2.py $(SAVE_DIR)/$(NAME)
	cp models/research/object_detection/exporter_main_v2.py $(SAVE_DIR)/$(NAME)
	touch $(SAVE_DIR)/$(NAME)/annotations/label_map.pbtxt