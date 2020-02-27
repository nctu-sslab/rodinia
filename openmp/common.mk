OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../common.mk

OMPFLAGS = -fopenmp -I/home/pschen/llvm/thesis/build-Debug/include

CFLAGS   += $(OMPFLAGS)
CXXFLAGS   += $(OMPFLAGS) -D__FUCK_FOR_THESIS__
LDLIBS   += $(OMPFLAGS)

ifdef AT
OFFLOAD=1
OFFLOADATFLAGS = -I/home/pschen/sslab/src-pschen/omp_offloading/include -DOMP_AT
CFLAGS += $(OFFLOADATFLAGS)
CXXFLAGS += $(OFFLOADATFLAGS)
endif

ifdef OFFLOAD
OMPOFFLOADFLAGS = -DOMP_OFFLOAD -fopenmp-targets=nvptx64
CXXFLAGS += $(OMPOFFLOADFLAGS)
CFLAGS += $(OMPOFFLOADFLAGS)
LDLIBS += $(OMPOFFLOADFLAGS)
endif

