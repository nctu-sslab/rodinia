include ../common.mk

EXE1  = kmeans
EXE2 = kmeans1D
EXE3 = kmeansCUDA
EXE4 = kmeansCUDA1D
EXE = $(EXE1) $(EXE2) $(EXE3) $(EXE4)
OBJS1 = cluster.o getopt.o kmeans.o kmeans_clustering.o
OBJS2 = cluster1D.o getopt.o kmeans1D.o kmeans_clustering1D.o
OBJS3 = cluster.o getopt.o kmeans.o kmeans_clusteringCUDA.o
OBJS4 = cluster1D.o getopt.o kmeans1D.o kmeans_clusteringCUDA1D.o
OBJS = $(OBJS1) $(OBJS2) $(OBJ3) $(OBJS4)

.PHONY: all
all: $(EXE)

kmeans_clusteringCUDA.o: kmeans_clusteringCUDA.cu
	nvcc $< -o $@ -c

kmeans_clusteringCUDA1D.o: kmeans_clustering1D.c
	cp $< kmeans_clustering1D.cu
	nvcc kmeans_clustering1D.cu -o $@ -DCUDA
	#rm kmeans_clustering1D.c

$(EXE1): $(OBJS1)

$(EXE2): $(OBJS2)

$(EXE3): $(OBJS3)
	clang++ -O2 -fopenmp -I/home/pschen/llvm/thesis/build-Debug/include $^ -lcuda -L/usr/local/cuda/lib64 -lcudart -o $@

$(EXE4): $(OBJS4)
	clang++ -O2 -fopenmp -I/home/pschen/llvm/thesis/build-Debug/include $^ -lcuda -L/usr/local/cuda/lib64 -lcudart -o $@

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
