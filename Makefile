RAJA_INSTALL_DIR := dist/RAJA
CXX := clang++
CXXFLAGS := -I$(RAJA_INSTALL_DIR)/include -I. -O3 -march=native
CPPFLAGS := -std=c++11 -fopenmp
LDFLAGS := $(RAJA_INSTALL_DIR)/lib/libRAJA.a

SRC := $(wildcard src/*.cpp)
OBJ := $(patsubst src/%.cpp,obj/%.o,$(SRC))
SRC := $(filter-out src/polybench_raja.cpp,$(SRC))
BIN := $(patsubst src/%.cpp,bin/%,$(SRC))

.PHONY: all clean setup

all : setup $(BIN)

setup :
	-@mkdir -p bin obj

$(BIN) : bin/% : obj/%.o obj/polybench_raja.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ -o $@

$(OBJ) : obj/%.o : src/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

clean :
	@-rm -vfr $(OBJ) $(BIN) obj bin
