RAJA_INSTALL_DIR := ../../dist/RAJA
CXX := clang++
CXXFLAGS := -I$(RAJA_INSTALL_DIR)/include -I. -O3 -march=native
CPPFLAGS := -std=c++11 -fopenmp
LDFLAGS := $(RAJA_INSTALL_DIR)/lib/libRAJA.a

SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
OBJ := $(addprefix obj/,$(OBJ))
SRC := $(filter-out polybench_raja.cpp,$(SRC))
BIN := $(SRC:.cpp=)
BIN := $(addprefix bin/,$(BIN))

.PHONY: all clean setup

all : setup $(BIN)

setup :
	-@mkdir -p bin obj

$(BIN) : bin/% : obj/%.o obj/polybench_raja.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ -o $@

$(OBJ) : obj/%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

clean :
	@-rm -vf $(OBJ) $(BIN)
