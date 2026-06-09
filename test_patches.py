import os
import unittest
import patch_submodules


class TestSubmodulePatches(unittest.TestCase):
    """
    Unit test suite to prevent regressions if/when ComfyUI or its custom nodes update.
    This suite checks that all target files exist, and that either the pristine target
    or the patched replacement exists in each target file.
    """

    def test_all_files_exist(self):
        """
        Verify that all files targeted by the patches exist on the filesystem.
        """
        for patch in patch_submodules.PATCHES:
            filepath = patch["filepath"]
            description = patch["description"]
            self.assertTrue(
                os.path.exists(filepath),
                msg=f"❌ Error: Targeted file '{filepath}' for '{description}' does not exist!"
            )

    def test_patch_signatures_present(self):
        """
        For each patch, assert that either the pristine target string or the
        already-applied replacement string exists in the target file.
        If neither is present, it indicates that the upstream source file structure
        has changed (e.g., due to an update) and the patch needs to be updated.
        """
        failed_patches = []
        for patch in patch_submodules.PATCHES:
            filepath = patch["filepath"]
            description = patch["description"]
            target = patch["target"]
            replacement = patch["replacement"]

            if not os.path.exists(filepath):
                continue  # Covered by test_all_files_exist

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            target_present = target in content
            replacement_present = replacement in content

            if not target_present and not replacement_present:
                failed_patches.append({
                    "filepath": filepath,
                    "description": description,
                    "reason": "Neither target nor replacement signatures were found in the file."
                })
            elif target_present and replacement_present:
                # If target is part of replacement, it might be both. That's fine as long as replacement is fully matching.
                pass

        if failed_patches:
            msg = "\n".join([
                f"🚨 REGRESSION DETECTED! Patch failed validation for '{p['description']}' in '{p['filepath']}':\n   Reason: {p['reason']}\n   -> Please update the target signature in 'patch_submodules.py' to match the updated file."
                for p in failed_patches
            ])
            self.fail(msg)

    def test_patch_applicability_dry_run(self):
        """
        Perform a dry-run check of patch applicability.
        """
        print("\n🧪 Running patch dry-run checks...")
        for patch in patch_submodules.PATCHES:
            filepath = patch["filepath"]
            description = patch["description"]
            target = patch["target"]
            replacement = patch["replacement"]

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if replacement in content:
                print(f"   [Dry-run] ✅ '{description}' is already successfully applied.")
            elif target in content:
                print(f"   [Dry-run] ⏳ '{description}' is NOT yet applied, but the pristine target signature is present and can be cleanly patched.")
            else:
                self.fail(f"❌ '{description}' cannot be applied to '{filepath}' because the target signature is missing.")


if __name__ == "__main__":
    unittest.main()
