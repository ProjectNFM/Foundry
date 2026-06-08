def test_package_reexports_public_api():
    from foundry.tasks import (
        CrossEntropyTaskLoss,
        MLPReadoutHead,
        MSETaskLoss,
        ReadoutHead,
        TargetExtractor,
        classification_metrics,
        regression_metrics,
        ssl_metrics,
    )

    assert TargetExtractor is not None
    assert ReadoutHead is not None
    assert CrossEntropyTaskLoss is not None
    assert MSETaskLoss is not None
    assert MLPReadoutHead is not None
    assert callable(classification_metrics)
    assert callable(regression_metrics)
    assert callable(ssl_metrics)
