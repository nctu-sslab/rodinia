SHELL=/bin/sh -ue

CC       = clang
CXX      = clang++

ifdef O0
CFLAGS   += -O0
CXXFLAGS += -O0
else
CFLAGS   += -O2
CXXFLAGS += -O2
endif

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
CFLAGS   += -DOUTPUT
endif

ifdef DEBUG
CFLAGS   += -g
CXXFLAGS += -g
endif

ifdef VERBOSE
CFLAGS   += -v
CXXFLAGS += -v
endif

# include Make.user relative to every active Makefile, exactly once
MAKEFILE_DIRS = $(foreach MAKEFILE,$(realpath $(MAKEFILE_LIST)), $(shell dirname $(MAKEFILE)))
$(foreach DIR,$(sort $(MAKEFILE_DIRS)),\
	$(eval -include $(DIR)/Make.user)\
)
