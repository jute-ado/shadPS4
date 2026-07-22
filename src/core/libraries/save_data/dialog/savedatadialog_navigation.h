// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Libraries::SaveData::Dialog {

enum class ListFocusAction {
    None,
    Restore,
    RestoreAndActivate,
};

constexpr ListFocusAction ChooseListFocusAction(bool first_render, bool has_navigation_id,
                                                bool confirm_pressed) {
    if (first_render) {
        return ListFocusAction::Restore;
    }
    if (has_navigation_id) {
        return ListFocusAction::None;
    }
    return confirm_pressed ? ListFocusAction::RestoreAndActivate : ListFocusAction::Restore;
}

} // namespace Libraries::SaveData::Dialog
