all: multiTrain multiPred
	
#.PHONY: multiTrain multiTrainHash multiPred

multiTrain:
	g++ -fopenmp -g -std=c++11 -O3 -o multiTrain multiTrain.cpp

multiTrainHash:	
	g++ -fopenmp -g -std=c++11 -O3 -o multiTrainHash multiTrain.cpp -DUSING_HASHVEC
	
multiPred:
	g++ -fopenmp -g -std=c++11 -O3 -o multiPred multiPred.cpp

clean:
	rm -f multiTrain
	rm -f multiTrainHash
	rm -f multiPred

#parameters

output_model=model
data_dir=
train_file=
heldout_file=
test_file=
misc=

.SECONDEXPANSION:

#multilabel datasets
LSHTCwiki_original: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.1" output_model="LSHTCwiki_original.model.limited_max_select_nodoubling.l01" misc="-d"

rcv1_regions:  examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout embeddings_file=$(base).embeddings test_file=$(base).test split_up_rate="-q 1" 

bibtex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout embeddings_file=$(base).embeddings test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1" 

scene2: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test early_terminate="-e 10"

synthetic: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

emotions: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1"

yeast: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test early_terminate="-e 10" speed_up_rate="-r 1" 

tmc: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1" 

Mediamill: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" speed_up_rate="-r 1" 

Eur-Lex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.001" early_terminate="-e 5"

#multiclass datasets
sector: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test


breast_cancer: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1"
structsvm: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

covtype: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1"

digits: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

aloi.bin: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.01"

aloi: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

Dmoz: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test split_up_rate="-q 3"

LSHTC1: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.01" split_up_rate="-q 3" early_terminate="-e 3"

imageNet: examples/$$@/	
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test split_up_rate="-q 3"

train_without_hash: multiTrain multiPred $(train_file) $(heldout_file) $(test_file) $(embeddings_file)
	rm -f dumps/*
	./multiTrain $(cost) $(lambda) $(solver) $(speed_up_rate) $(early_terminate) $(max_iter) $(split_up_rate) $(max_select) $(post_train_iter) $(sample_option) $(misc) -h $(heldout_file) -T $(embeddings_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(embeddings_file) $(heldout_file) $(output_model) $(top)
	./multiPred $(embeddings_file) $(test_file) $(output_model) $(top)
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(heldout_file) $(output_model).p $(top)
	./multiPred $(test_file) $(output_model).p $(top)
endif

train_with_hash: multiTrainHash multiPred $(train_file) $(heldout_file) $(test_file) $(embeddings_file)
	rm -f dumps/*
	./multiTrainHash $(cost) $(lambda) $(solver) $(speed_up_rate) $(early_terminate) $(max_iter) $(split_up_rate) $(max_select) $(post_train_iter) $(sample_option) $(misc) -h $(heldout_file) -T $(embeddings_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(embeddings_file) $(heldout_file) $(output_model) $(top)
	./multiPred $(embeddings_file) $(test_file) $(output_model) $(top)
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(heldout_file) $(output_model).p $(top)
	./multiPred $(test_file) $(output_model).p $(top)
endif

examples/%:
	make construct -C examples/ dataset=$(notdir $@)

