from __future__ import annotations

from collections import defaultdict

from engine.models import all_registered_models, get_model_family, get_model_repo_id


def main() -> None:
    families: dict[str, list[str]] = defaultdict(list)

    for model in all_registered_models(newest_first=False):
        families[get_model_family(model)].append(model)

    if not families:
        print("No models are currently registered.")
        return

    print("Registered models:")
    for family in sorted(families):
        print(f"\n{family}:")
        for model in families[family]:
            print(f"  - {model} ({get_model_repo_id(model)})")


if __name__ == "__main__":
    main()
