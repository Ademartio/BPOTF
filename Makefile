
MODULE_NAME		:= bpbp
MODULE_DIR		:= ./module
INC_DIR			:= ./src
PYBIND11_INC 	:= $(shell python3 -m pybind11 --includes)
MOD_EXT_SUFIX	:= $(shell python3-config --extension-suffix)
PY_LIBS			:= $(shell python3-config --ldflags)
SRCS				:= src/BPOTF/OBPOTF.cpp \
						src/DisjointSet/DisjointSet.cpp \
						src/py11_iface.cpp 

ENVIRONMENT := $(shell uname -s | sed -E 's/^([^-]*).*$$/\1/g')
ifeq ($(OS),Windows_NT)
	CCFLAGS += -D WIN64
	ifeq ($(ENVIRONMENT),CYGWIN_NT)
		STD = gnu++11
	else
		STD = c++11
	endif
else
	CCFLAGS += -D Linux
	STD = gnu++11
	STD = c++2a
endif
		


bpbp_module: 
	[ -d $(MODULE_DIR) ] || mkdir -p $(MODULE_DIR)
	g++ -O3 -Wall -shared -std=${STD} ${CCFLAGS} -fPIC -I${INC_DIR} ${PYBIND11_INC} ${SRCS} -o ${MODULE_DIR}/${MODULE_NAME}${MOD_EXT_SUFIX} ${PY_LIBS}
	cp -r ${MODULE_DIR} ./reference

clean:
	rm -rf $(MODULE_DIR)
