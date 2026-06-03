MAKEFLAGS += -s

CROSS_COMPILE:=/home/ljhong/GCC/mips-gcc540-glibc222-r3.3.7.mxu2.mem03/bin/mips-linux-gnu-
# CROSS_COMPILE:=/home/ljhong/GCC/mips-gcc472-glibc216-r2.4.2.npu/bin/mips-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)gcc-ar
STRIP = $(CROSS_COMPILE)strip

# muclibc, mglibc
LIBTYPE ?= muclibc
# nv12, rgba/bgra
INTYPE ?= rgba
# static, dynamic
LIBSTATUS ?= dynamic

HOME_LIB_PATH = ../../../../../InferenceKit/uranus/tnna/

DEF_OPTTIONS= -std=c++14 -mfp32 -Wall -EL -O3 -Os -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm -fPIC -flto -Wl,-Map=object.map,--cref,--gc-section #For 540
# DEF_OPTTIONS= -std=c++11 -mfp32 -Wall -EL -O3 -Os -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm -fPIC -flto -Wl,-Map=object.map,--cref,--gc-section #For 472

CXXFLAGS := $(DEFINE_OPT) -ffunction-sections -fdata-sections -fno-exceptions

ifeq ($(INTYPE), nv12)
	CXXFLAGS += -DURANUS_NV12
endif

ifeq ($(LIBTYPE), muclibc)
	CXXFLAGS += -muclibc
	LIB_PATH = $(HOME_LIB_PATH)/540/lib/uclibc
    ifeq ($(LIBSTATUS), dynamic)
	    TARGET = load_mgk_uclibc
    else
	    TARGET = load_mgk_uclibc_static
    endif
else
	LIB_PATH = $(HOME_LIB_PATH)/540/lib/glibc
    ifeq ($(LIBSTATUS), dynamic)
	    TARGET = load_mgk_glibc
    else
	    TARGET = load_mgk_glibc_static
    endif
endif 

INCLUDES := -I$(HOME_LIB_PATH) -I$(HOME_LIB_PATH)/include/

# 使用-L指定库路径，-l指定库名称
LDFLAGS := -L$(LIB_PATH) -Wl,-rpath=$(LIB_PATH) -Wl,-gc-sections -fno-exceptions -Wl,-export-dynamic
LIBS := -luranus_tensor -luranus_core -luranus_extend -laie

# 静态库仍然需要直接指定文件路径
STATIC_LIBS := $(LIB_PATH)/liburanus_tensor.a $(LIB_PATH)/liburanus_core.a $(LIB_PATH)/liburanus_extend.a $(LIB_PATH)/libaie.a

OBJS := inference_yolov5s.o
# OBJS := inference_yolov5s_nv12.o

%.o:%.cpp
	$(CPP) $(DEF_OPTTIONS) $(INCLUDES) $(CXXFLAGS) -o $@ $< -c

$(TARGET):$(OBJS)
	@echo "Building $(TARGET) with $(LIBSTATUS) libraries..."
	@echo "Library path: $(LIB_PATH)"
ifeq ($(LIBSTATUS), dynamic)
	# 检查动态库是否存在
	@if [ ! -f "$(LIB_PATH)/liburanus_tensor.so" ]; then \
		echo "Error: Dynamic libraries not found in $(LIB_PATH)"; \
		echo "Available files:"; \
		ls -la $(LIB_PATH)/liburanus_* $(LIB_PATH)/libaie* 2>/dev/null || echo "No library files found"; \
		exit 1; \
	fi
	$(CPP) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)
else
	# 检查静态库是否存在
	@if [ ! -f "$(LIB_PATH)/liburanus_tensor.a" ]; then \
		echo "Error: Static libraries not found in $(LIB_PATH)"; \
		echo "Available files:"; \
		ls -la $(LIB_PATH)/liburanus_* $(LIB_PATH)/libaie* 2>/dev/null || echo "No library files found"; \
		exit 1; \
	fi
	$(CPP) $(CXXFLAGS) $^ -o $@ -Wl,-whole-archive $(STATIC_LIBS) -Wl,-no-whole-archive -ldl $(LDFLAGS)
endif

# 检查库文件的目标
check-libs:
	@echo "Checking libraries in $(LIB_PATH):"
	@ls -la $(LIB_PATH)/liburanus_* $(LIB_PATH)/libaie* 2>/dev/null || echo "No library files found"
	@echo ""
	@echo "Dynamic libraries:"
	@ls -la $(LIB_PATH)/*.so 2>/dev/null || echo "No .so files found"
	@echo ""
	@echo "Static libraries:"
	@ls -la $(LIB_PATH)/*.a 2>/dev/null || echo "No .a files found"

all : $(TARGET)

.PHONY: all clean check-libs
clean:
	rm -rf $(OBJS) $(TARGET) object.map
