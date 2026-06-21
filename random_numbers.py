import random
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Genera e analizza numeri casuali.")
        print("Utilizzo: python random_numbers.py [intervallo] [quantita]")
        print("  intervallo - valore massimo (default: 100)")
        print("  quantita   - quanti numeri generare (default: 10)")
        return

    max_val = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    numeri = [random.randint(1, max_val) for _ in range(n)]

    print(f"{n} numeri casuali tra 1 e {max_val}:")
    print("  ", " ".join(map(str, numeri)))
    print(f"  Media  : {sum(numeri) / n:.2f}")
    print(f"  Min    : {min(numeri)}")
    print(f"  Max    : {max(numeri)}")
    print(f"  Somma  : {sum(numeri)}")
    print(f"  Unici  : {len(set(numeri))}")


if __name__ == "__main__":
    main()
