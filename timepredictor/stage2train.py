from configs.default import Config, set_seed
from engine.trainer import run_improved_model


def main():
    print(">>> Improved CNN-BiLSTM with CBAM-BRA")
    set_seed(Config.seed)
    result = run_improved_model(seed=Config.seed, save_artifacts=True)
    print(result["name"], "completed.")


if __name__ == "__main__":
    main()

