CC = gcc
CFLAGS = -O3 -Wall -fopenmp -std=c99
LDFLAGS = -lm
TARGET = out
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

CYAN = \033[0;36m
YELLOW = \033[0;33m
GREEN = \033[0;32m
RED = \033[0;31m
RESET = \033[0m
test_local: $(TARGET)
	@printf "${YELLOW}%10sTest local${RESET}\n"
	./$(TARGET) 127 1.0 71 1
	@echo ""
	./$(TARGET) 127 1.0 71 2
	@echo ""
	./$(TARGET) 127 1.0 71 4
	@echo ""
	./$(TARGET) 127 1.0 71 8

clean:
	rm -r $(TARGET)

.PHONY: all test_local
