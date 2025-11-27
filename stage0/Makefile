CC = gcc
IBM_CC = xlc_r

OPTIMFLAG = -O2
FFLAGS = -fopenmp 
CFLAGS = -Wall -std=c99
OMPFLAGS = -mcpu=power8 -qsmp=omp
LDFLAGS = -lm

TARGET = out
TARGET_HOME = home

SRC = main.c

ifdef ERRORS
  CFLAGS += -DCOMPUTE_ERRORS
endif

ifdef REPORT
  CFLAGS += -DCOMPUTE_ERRORS -DLOG_ERRORS
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(IBM_CC) $(OMPFLAGS) -o $(TARGET) $(SRC) $(OPTIMFLAG)

$(TARGET_HOME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(FFLAGS) -o $(TARGET_HOME) $(SRC) $(OPTIMFLAG)

CYAN = \033[0;36m
YELLOW = \033[0;33m
GREEN = \033[0;32m
RED = \033[0;31m
RESET = \033[0m

n128_01: $(TARGET_HOME)
	@printf "${YELLOW}%10sTest n128 0.1${RESET}\n"
	@printf "${CYAN}%sNp = 1${RESET}\n"
	./$(TARGET_HOME) 127 1.0 71 1
	@echo ""
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET_HOME) 127 1.0 71 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET_HOME) 127 1.0 71 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET_HOME) 127 1.0 71 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET_HOME) 127 1.0 71 16
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n256_01: $(TARGET_HOME)
	@printf "${YELLOW}%10sTest n256 0.1${RESET}\n"
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET_HOME) 255 1.0 141 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET_HOME) 255 1.0 141 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET_HOME) 255 1.0 141 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET_HOME) 255 1.0 141 16
	@echo ""
	@printf "${CYAN}%sNp = 32${RESET}\n"
	./$(TARGET_HOME) 255 1.0 141 32
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n128_pi: $(TARGET_HOME)
	@printf "${YELLOW}%10sTest n128 pi${RESET}\n"
	@printf "${CYAN}%sNp = 1${RESET}\n"
	./$(TARGET_HOME) 127 3.14159265359 71 1
	@echo ""
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET_HOME) 127 3.14159265359 71 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET_HOME) 127 3.14159265359 71 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET_HOME) 127 3.14159265359 71 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET_HOME) 127 3.14159265359 71 16
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi

n256_pi: $(TARGET_HOME)
	@printf "${YELLOW}%10sTest n256 pi${RESET}\n"
	@printf "${CYAN}%sNp = 2${RESET}\n"
	./$(TARGET_HOME) 255 3.14159265359 141 2
	@echo ""
	@printf "${CYAN}%sNp = 4${RESET}\n"
	./$(TARGET_HOME) 255 3.14159265359 141 4
	@echo ""
	@printf "${CYAN}%sNp = 8${RESET}\n"
	./$(TARGET_HOME) 255 3.14159265359 141 8
	@echo ""
	@printf "${CYAN}%sNp = 16${RESET}\n"
	./$(TARGET_HOME) 255 3.14159265359 141 16
	@echo ""
	@printf "${CYAN}%sNp = 32${RESET}\n"
	./$(TARGET_HOME) 255 3.14159265359 141 32
	@if [ -n "$(wildcard *.txt)" ]; then echo "Plotting errors" && ./plot.sh *.txt; fi
	
clean:
	rm -rf $(TARGET) $(TARGET_HOME) *.txt *.png

.PHONY: all test_local
