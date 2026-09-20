"""Roll the serving alias back to a previously registered model version.

Rollback moves the *alias*, not the files: MLflow keeps every registered
version's artifacts, so the served bundle changes as soon as serving
re-resolves ``champion``. This system resolves the alias once at startup
(a per-request registry lookup would put a network call on the user-facing
path for a value that changes weekly), so a rollback is: move the alias,
restart the serving process.

Because the payout surrogate, the TabPFN cache reference, the fallback table
and the name index are all logged as artifacts of the *same run* as the
classifier, one alias move rolls the whole bundle back together — there is no
window where a new classifier is paired with an old payout model.
"""
from __future__ import annotations

import argparse

from mlflow import MlflowClient

from bl_ranking.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to-version", default=None, help="Explicit registered version to promote to champion")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    settings = get_settings()
    name = args.model_name or settings.mlflow_registry_model_name_classifier
    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_registry_uri or None
    )

    current = client.get_model_version_by_alias(name, "champion")
    print(f"current champion: {name} v{current.version}")

    if args.to_version:
        target = str(args.to_version)
    else:
        try:
            target = str(client.get_model_version_by_alias(name, "previous").version)
        except Exception as exc:
            raise SystemExit(
                "no 'previous' alias is set — pass --to-version explicitly, or list versions in the MLflow UI"
            ) from exc

    if target == str(current.version):
        raise SystemExit(f"target version {target} is already champion — nothing to do")

    client.set_registered_model_alias(name, "previous", current.version)
    client.set_registered_model_alias(name, "champion", target)
    print(f"rolled back: {name} champion {current.version} -> {target} (old champion kept as 'previous')")
    print("restart the serving process to pick up the new champion.")


if __name__ == "__main__":
    main()
