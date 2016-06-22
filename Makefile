RAJA_INSTALL_DIR := dist/RAJA
CXX := clang++
OPTS ?= -O3 -march=native
CXXFLAGS := -I$(RAJA_INSTALL_DIR)/include -I./include $(OPTS)
CPPFLAGS := -std=c++11 -fopenmp=libomp -MMD -Wall -Wextra -pedantic
LDLIBS := $(RAJA_INSTALL_DIR)/lib/libRAJA.a -lrt
ARFLAGS := r

INSTALLPREFIX := dist/PolyBench
PREFIX := build/PolyBench

SRCDIR := src
OBJDIR := $(PREFIX)/obj
LIBDIR := $(INSTALLPREFIX)/lib
BINDIR := $(INSTALLPREFIX)/bin

LIBSRCS := PolyBenchKernel.cpp
LIBSRC := $(patsubst %,$(SRCDIR)/%,$(LIBSRCS))

SRC := $(wildcard $(SRCDIR)/*.cpp)
DEPS := $(patsubst $(SRCDIR)%.cpp,$(OBJDIR)/%.d,$(SRC))

SRC := $(filter-out $(LIBSRC),$(SRC))

OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRC))
LIBOBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(LIBSRC))

LIB := $(LIBDIR)/libPolyBench.a
BIN := $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%,$(SRC))

.PHONY: all clean setup

all : setup $(BIN)

setup :
	@-mkdir -p $(BINDIR) $(OBJDIR) $(LIBDIR)

$(BIN) : $(BINDIR)/% : $(OBJDIR)/%.o $(LIB)
	@echo "[LINK] $@"
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

$(OBJ) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@echo "[COMPILE] $< --> $@"
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(LIBOBJ) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@echo "[COMPILE] $< --> $@"
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(LIB) : $(LIBOBJ)
	@echo "[ARCHIVE] ($^) --> $@"
	@$(AR) $(ARFLAGS) $@ $^ >& /dev/null

lib : $(LIBDIR) $(LIB)

clean :
	@-rm -fr $(OBJDIR) $(BINDIR) $(LIBDIR)

DIRS := $(OBJDIR) $(BINDIR) $(LIBDIR)

dirs : $(DIRS)

$(OBJDIR) :
	@-mkdir -p $@

$(BINDIR) :
	@-mkdir -p $@

$(LIBDIR) :
	@-mkdir -p $@

-include $(DEPS)
