
# output
BIN              := runner

# flags
INCLUDES         := -I.
LIBS             := -lstdc++
CFLAGS           := -std=c++11
LDFLAGS          :=

# compilers
CC               := gcc
LINKER           := gcc

# source files
C_SOURCES        := $(wildcard *.c)
CPP_SOURCES      := $(wildcard *.cpp)
HEADERS          := $(wildcard *.h)
C_OBJS           := $(patsubst %.c, %.o, $(C_SOURCES))
CPP_OBJS         := $(patsubst %.cpp, %.o, $(CPP_SOURCES))

# OpenCV support
CFLAGS		+= `/usr/local/bin/pkg-config --cflags opencv4`
LDFLAGS		+= `/usr/local/bin/pkg-config --libs opencv4`
LIBS		+= `/usr/local/bin/pkg-config --libs opencv4`


$(BIN): clean $(C_OBJS) $(CPP_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS) $(INCLUDES)

$(CPP_OBJS): $(CPP_SOURCES) $(HEADERS)
	$(CC) -c $(CPP_SOURCES) $(CFLAGS) $(INCLUDES)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) *.o