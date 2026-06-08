import torch

from foundry.models.readout import ReadoutRouter
from foundry.tasks.heads import ReadoutHead


class TestReadoutRouterSingleTask:
    def test_ignores_task_index_and_routes_all_embeddings(self):
        embed_dim, output_dim = 16, 4
        head = ReadoutHead(embed_dim=embed_dim, output_dim=output_dim)
        router = ReadoutRouter({"task_a": head})
        embs = torch.randn(6, embed_dim)
        task_index = torch.tensor([0, 1, 2, 3, 4, 5])

        outputs = router(embs, task_index=task_index)

        assert set(outputs.keys()) == {"task_a"}
        assert outputs["task_a"].shape == (6, output_dim)
        expected = head(embs)
        assert torch.allclose(outputs["task_a"], expected)


class TestReadoutRouterMultiTask:
    def test_routes_embeddings_by_task_index(self):
        embed_dim = 8
        head_alpha = ReadoutHead(embed_dim=embed_dim, output_dim=2)
        head_beta = ReadoutHead(embed_dim=embed_dim, output_dim=3)
        router = ReadoutRouter({"alpha": head_alpha, "beta": head_beta})
        embs = torch.randn(4, embed_dim)
        task_index = torch.tensor([0, 0, 1, 1])

        outputs = router(embs, task_index=task_index)

        assert set(outputs.keys()) == {"alpha", "beta"}
        assert torch.allclose(outputs["alpha"], head_alpha(embs[:2]))
        assert torch.allclose(outputs["beta"], head_beta(embs[2:]))

    def test_skips_tasks_with_no_tokens_in_batch(self):
        embed_dim = 8
        head_alpha = ReadoutHead(embed_dim=embed_dim, output_dim=2)
        head_beta = ReadoutHead(embed_dim=embed_dim, output_dim=3)
        router = ReadoutRouter({"alpha": head_alpha, "beta": head_beta})
        embs = torch.randn(3, embed_dim)
        task_index = torch.tensor([0, 0, 0])

        outputs = router(embs, task_index=task_index)

        assert set(outputs.keys()) == {"alpha"}
        assert torch.allclose(outputs["alpha"], head_alpha(embs))


class TestReadoutRouterTaskIndices:
    def test_task_index_for_uses_sorted_name_order(self):
        embed_dim = 4
        heads = {
            "zebra": ReadoutHead(embed_dim=embed_dim, output_dim=1),
            "alpha": ReadoutHead(embed_dim=embed_dim, output_dim=1),
            "middle": ReadoutHead(embed_dim=embed_dim, output_dim=1),
        }
        router = ReadoutRouter(heads)

        assert router.task_index_for("alpha") == 0
        assert router.task_index_for("middle") == 1
        assert router.task_index_for("zebra") == 2
        assert router.num_tasks == 3


class TestReadoutRouterParameters:
    def test_exposes_all_head_parameters(self):
        embed_dim = 6
        head_a = ReadoutHead(embed_dim=embed_dim, output_dim=2)
        head_b = ReadoutHead(embed_dim=embed_dim, output_dim=3)
        router = ReadoutRouter({"task_a": head_a, "task_b": head_b})

        router_param_ids = {id(p) for p in router.parameters()}
        head_param_ids = {id(p) for p in head_a.parameters()} | {
            id(p) for p in head_b.parameters()
        }

        assert head_param_ids.issubset(router_param_ids)


class TestReadoutRouterGradients:
    def test_gradients_flow_through_router(self):
        embed_dim = 8
        head_alpha = ReadoutHead(embed_dim=embed_dim, output_dim=2)
        head_beta = ReadoutHead(embed_dim=embed_dim, output_dim=3)
        router = ReadoutRouter({"alpha": head_alpha, "beta": head_beta})
        embs = torch.randn(4, embed_dim, requires_grad=True)
        task_index = torch.tensor([0, 0, 1, 1])

        outputs = router(embs, task_index=task_index)
        loss = outputs["alpha"].sum() + outputs["beta"].sum()
        loss.backward()

        assert embs.grad is not None
        assert head_alpha.projection.weight.grad is not None
        assert head_beta.projection.weight.grad is not None
