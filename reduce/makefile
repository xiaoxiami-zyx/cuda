# Makefile for CUDA project

# Compiler
NVCC = nvcc
CCBIN = icpx

# Flags
CFLAGS = -O3 -arch=sm_120 -ccbin=$(CCBIN) -Xcompiler=-Wall -allow-unsupported-compiler

# Source files
SRCS = main.cu
# Object files
OBJS = $(SRCS:.cu=.o)
# Executable name
TARGET = reduce

# Default target
all: $(TARGET)
# Link object files to create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^
# Compile source files to object files
%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@
# Clean up object files and executable
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
