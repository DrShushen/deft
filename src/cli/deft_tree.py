import hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg) -> None:
    # Lazy import so `deft-tree --help` / `--cfg` work with only hydra installed
    # (the heavy src.* / ML stack is pulled only when training actually runs).
    from src.cli.run_deft_tree import run_experiment

    run_experiment(cfg)


if __name__ == "__main__":
    main()
