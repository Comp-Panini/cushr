CXX      ?= g++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra -Wpedantic
INCLUDES  = -Iinclude -Ivendor
LDLIBS    = -lz

LIB_SRC  = src/lattice.cpp src/scorer.cpp src/decoder.cpp \
            src/metrics.cpp src/json_io.cpp vendor/cnpy.cpp
LIB_OBJ  = $(LIB_SRC:.cpp=.o)

TEST_SRC = tests/test_main.cpp tests/test_lattice.cpp \
            tests/test_decoder.cpp tests/test_scorer.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)

.PHONY: all test clean

all: cushr_evaluate

cushr_evaluate: $(LIB_OBJ) apps/evaluate.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

cushr_tests: $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

test: cushr_tests
	./cushr_tests

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(LIB_OBJ) $(TEST_OBJ) apps/evaluate.o cushr_evaluate cushr_tests
