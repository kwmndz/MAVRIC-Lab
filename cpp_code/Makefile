CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -pedantic

TARGET = test.exe
SRCS = DBF_tests.cpp DynamicBayesianFiltering.cpp
HEADERS = DynamicBayesianFiltering.hpp

$(TARGET): $(SRCS)
    $(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
    rm -f $(TARGET)