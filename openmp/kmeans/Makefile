include ../common.mk

EXE1  = kmeans
EXE2 = kmeans1D
EXE = $(EXE1) $(EXE2)
OBJS1 = cluster.o getopt.o kmeans.o kmeans_clustering.o
OBJS2 = cluster1D.o getopt.o kmeans1D.o kmeans_clustering1D.o
OBJS = $(OBJS1) $(OBJS2)

.PHONY: all
all: $(EXE)

$(EXE1): $(OBJS1)
$(EXE2): $(OBJS2)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
