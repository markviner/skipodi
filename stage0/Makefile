CC = gcc
CFLAGS = -O3 -Wall -fopenmp -std=c99
LDFLAGS = -lm
TARGET = out
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

ifdef REPORT
  CFLAGS += -DSOLVE_WITH_ERROR
endif


CYAN = \033[0;36m
YELLOW = \033[0;33m
GREEN = \033[0;32m
RED = \033[0;31m
RESET = \033[0m
test_local: $(TARGET)
	@printf "${YELLOW}%10sTest local${RESET}\n"
	./$(TARGET) 127 1.0 141 1
	@echo ""
	./$(TARGET) 127 1.0 141 2
	@echo ""
	./$(TARGET) 127 1.0 141 4
	@echo ""
	./$(TARGET) 127 1.0 141 8

n128_01: $(TARGET)
	@printf "${YELLOW}%10sTest n128 0.1${RESET}\n"
	@printf "${CYAN}%sNp = 1${RESET}\n"
	./$(TARGET) 127 1.0 71 1
	@echo ""
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET) 127 1.0 71 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET) 127 1.0 71 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET) 127 1.0 71 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET) 127 1.0 71 16
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n256_01: $(TARGET)
	@printf "${YELLOW}%10sTest n256 0.1${RESET}\n"
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET) 255 1.0 141 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET) 255 1.0 141 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET) 255 1.0 141 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET) 255 1.0 141 16
	@echo ""
	@printf "${CYAN}%sNp = 32${RESET}\n"
	./$(TARGET) 255 1.0 141 32
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n128_pi: $(TARGET)
	@printf "${YELLOW}%10sTest n128 pi${RESET}\n"
	@printf "${CYAN}%sNp = 1${RESET}\n"
	./$(TARGET) 127 3.14159265359 71 1
	@echo ""
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET) 127 3.14159265359 71 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET) 127 3.14159265359 71 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET) 127 3.14159265359 71 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET) 127 3.14159265359 71 16
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n256_pi: $(TARGET)
	@printf "${YELLOW}%10sTest n256 pi${RESET}\n"
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET) 255 3.14159265359 141 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET) 255 3.14159265359 141 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET) 255 3.14159265359 141 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET) 255 3.14159265359 141 16
	@echo ""
	@printf "${CYAN}%sNp = 32${RESET}\n"
	./$(TARGET) 255 3.14159265359 141 32
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi
	
clean:
	rm -r $(TARGET) *.txt *.png

.PHONY: all test_local
