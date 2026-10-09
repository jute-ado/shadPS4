import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RASTERIZER = ROOT / "src" / "video_core" / "renderer_vulkan" / "vk_rasterizer.cpp"


def _function_body(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


class ImageDescriptorFinalizationOrderContract(unittest.TestCase):
    def setUp(self) -> None:
        self.source = RASTERIZER.read_text(encoding="utf-8")

    def test_opt_in_early_policy_is_explicit(self) -> None:
        self.assertIn("EarlyImageBindingFinalizationEnabled", self.source)
        self.assertIn("SHADPS4_EARLY_IMAGE_BINDING_FINALIZATION", self.source)

    def test_direct_draw_can_finalize_before_waiting_buffer_prepare(self) -> None:
        body = _function_body(
            self.source,
            "void Rasterizer::Draw(bool is_indexed, u32 index_offset)",
            "void Rasterizer::DrawIndirect(",
        )
        early = body.index("if (early_image_binding_finalization &&")
        prepare = body.index("PrepareDrawBuffersThenBindCommandState")
        callback = body.index("if (!early_image_binding_finalization)")
        self.assertLess(early, prepare)
        self.assertLess(body.index("BeginRendering", early), prepare)
        self.assertLess(body.index("FinalizeImageDescriptors", early), prepare)
        self.assertGreater(callback, prepare)

    def test_indirect_draw_can_finalize_before_waiting_buffer_prepare(self) -> None:
        body = _function_body(
            self.source,
            "void Rasterizer::DrawIndirect(",
            "void Rasterizer::DispatchDirect()",
        )
        early = body.index("if (early_image_binding_finalization &&")
        prepare = body.index("PrepareDrawBuffersThenBindCommandState")
        callback = body.index("if (!early_image_binding_finalization)")
        self.assertLess(early, prepare)
        self.assertLess(body.index("BeginRendering", early), prepare)
        self.assertLess(body.index("FinalizeImageDescriptors", early), prepare)
        self.assertGreater(callback, prepare)

    def test_image_descriptor_starts_with_live_view_for_early_publication(self) -> None:
        """An early-publication experiment must never publish a null image view."""
        bind = _function_body(
            self.source,
            "void Rasterizer::BindTextures(",
            "bool Rasterizer::RebindPendingImageBindings()",
        )
        find_view = bind.index("FindTexture(image_id, desc)")
        eager_view = bind.index(
            "image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,",
            find_view,
        )
        self.assertGreater(eager_view, find_view)


if __name__ == "__main__":
    unittest.main()
