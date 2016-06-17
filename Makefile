RAJA_INSTALL_DIR := dist/RAJA
CXX := clang++
CXXFLAGS := -I$(RAJA_INSTALL_DIR)/include -I./include -O3 -march=native
CPPFLAGS := -std=c++11 -fopenmp
LDFLAGS := $(RAJA_INSTALL_DIR)/lib/libRAJA.a

INSTALLPREFIX := dist/PolyBench
PREFIX := build/PolyBench

SRCDIR := src
OBJDIR := $(PREFIX)/obj
LIBDIR := $(INSTALLPREFIX)/lib
BINDIR := $(INSTALLPREFIX)/bin

LIBSRC := $(SRCDIR)/polybench_raja.cpp
SRC := $(wildcard $(SRCDIR)/*.cpp)
SRC := $(filter-out $(LIBSRC),$(SRC))

OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRC))
LIBOBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(LIBSRC))

LIB := $(LIBDIR)/libPolyBench.a
BIN := $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%,$(SRC))

.PHONY: all clean setup

all : setup $(BIN)

setup :
	-@mkdir -p $(BINDIR) $(OBJDIR) $(LIBDIR)

$(BIN) : $(BINDIR)/% : $(OBJDIR)/%.o $(LIB)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ -o $@

$(OBJ) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(LIBOBJ) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(LIB) : $(LIBOBJ)
	$(AR) $(ARFLAGS) $@ $<

clean :
	@-rm -vfr $(OBJDIR) $(BINDIR) $(LIBDIR)
