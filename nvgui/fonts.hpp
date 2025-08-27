/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Contains additional fonts for icons and monospace text.
// Example:
// ```
// ImGui::PushFont(ImGuiH::getIconicFont());
// ImGui::Button(ImGuiH::icon_account_login);
// ImGui::PopFont();
// ```

// To use the icons merged into the default font:
//
//	std::string buttonLabel = std::string("Login " + ICON_MS_LOGIN);
//	ImGui::Button(buttonLabel.c_str());
//  ImGui::Button(fmt::format("Login {}", ICON_MS_LOGIN).c_str());
//	ImGui::Button(ICON_MS_LOGIN);


#pragma once

#include "IconsMaterialSymbols.h"

struct ImFont;

namespace nvgui {

void    addDefaultFont(float fontSize = 15.F, bool appendIcons = true);  // Initializes the default font.
ImFont* getDefaultFont();                                                // Returns the default font.
void    addIconicFont(float fontSize = 15.F);                            // Initializes the iconic font.
ImFont* getIconicFont();                                                 // Returns the iconic font
void    showDemoIcons();                                                 // Show all icons in a separated window
void    addMonospaceFont(float fontSize = 15.F);                         // Initializes the monospace font.
ImFont* getMonospaceFont();                                              // Returns the monospace font


[[maybe_unused]] static const char* icon_account_login            = (char*)u8"\ue000";  // ICON_MS_LOGIN
[[maybe_unused]] static const char* icon_account_logout           = (char*)u8"\ue001";  // ICON_MS_LOGOUT
[[maybe_unused]] static const char* icon_action_redo              = (char*)u8"\ue002";  // ICON_MS_REDO
[[maybe_unused]] static const char* icon_action_undo              = (char*)u8"\ue003";  // ICON_MS_UNDO
[[maybe_unused]] static const char* icon_align_center             = (char*)u8"\ue004";  // ICON_MS_ALIGN_CENTER
[[maybe_unused]] static const char* icon_align_left               = (char*)u8"\ue005";  // ICON_MS_ALIGN_LEFT
[[maybe_unused]] static const char* icon_align_right              = (char*)u8"\ue006";  // ICON_MS_ALIGN_RIGHT
[[maybe_unused]] static const char* icon_aperture                 = (char*)u8"\ue007";  // ICON_MS_CAMERA
[[maybe_unused]] static const char* icon_arrow_bottom             = (char*)u8"\ue008";  // ICON_MS_ARROW_DOWNWARD
[[maybe_unused]] static const char* icon_arrow_circle_bottom      = (char*)u8"\ue009";  // ICON_MS_ARROW_CIRCLE_DOWN
[[maybe_unused]] static const char* icon_arrow_circle_left        = (char*)u8"\ue00A";  // ICON_MS_ARROW_CIRCLE_LEFT
[[maybe_unused]] static const char* icon_arrow_circle_right       = (char*)u8"\ue00B";  // ICON_MS_ARROW_CIRCLE_RIGHT
[[maybe_unused]] static const char* icon_arrow_circle_top         = (char*)u8"\ue00C";  // ICON_MS_ARROW_CIRCLE_UP
[[maybe_unused]] static const char* icon_arrow_left               = (char*)u8"\ue00D";  // ICON_MS_ARROW_BACK
[[maybe_unused]] static const char* icon_arrow_right              = (char*)u8"\ue00E";  // ICON_MS_ARROW_FORWARD
[[maybe_unused]] static const char* icon_arrow_thick_bottom       = (char*)u8"\ue00F";  // ICON_MS_ARROW_DOWNWARD_ALT
[[maybe_unused]] static const char* icon_arrow_thick_left         = (char*)u8"\ue010";  // ICON_MS_ARROW_BACK_ALT
[[maybe_unused]] static const char* icon_arrow_thick_right        = (char*)u8"\ue011";  // ICON_MS_ARROW_FORWARD_ALT
[[maybe_unused]] static const char* icon_arrow_thick_top          = (char*)u8"\ue012";  // ICON_MS_ARROW_UPWARD_ALT
[[maybe_unused]] static const char* icon_arrow_top                = (char*)u8"\ue013";  // ICON_MS_ARROW_UPWARD
[[maybe_unused]] static const char* icon_audio                    = (char*)u8"\ue014";  // ICON_MS_HEARING
[[maybe_unused]] static const char* icon_audio_spectrum           = (char*)u8"\ue015";  // ICON_MS_EQUALIZER
[[maybe_unused]] static const char* icon_badge                    = (char*)u8"\ue016";  // ICON_MS_LICENSE
[[maybe_unused]] static const char* icon_ban                      = (char*)u8"\ue017";  // ICON_MS_BLOCK
[[maybe_unused]] static const char* icon_bar_chart                = (char*)u8"\ue018";  // ICON_MS_BAR_CHART
[[maybe_unused]] static const char* icon_basket                   = (char*)u8"\ue019";  // ICON_MS_SHOPPING_BASKET
[[maybe_unused]] static const char* icon_battery_empty            = (char*)u8"\ue01A";  // ICON_MS_BATTERY_ALERT
[[maybe_unused]] static const char* icon_battery_full             = (char*)u8"\ue01B";  // ICON_MS_BATTERY_FULL
[[maybe_unused]] static const char* icon_beaker                   = (char*)u8"\ue01C";  // ICON_MS_SCIENCE
[[maybe_unused]] static const char* icon_bell                     = (char*)u8"\ue01D";  // ICON_MS_NOTIFICATIONS
[[maybe_unused]] static const char* icon_bluetooth                = (char*)u8"\ue01E";  // ICON_MS_BLUETOOTH
[[maybe_unused]] static const char* icon_bold                     = (char*)u8"\ue01F";  // ICON_MS_BOLD
[[maybe_unused]] static const char* icon_bolt                     = (char*)u8"\ue020";  // ICON_MS_FLASH_ON
[[maybe_unused]] static const char* icon_book                     = (char*)u8"\ue021";  // ICON_MS_BOOK
[[maybe_unused]] static const char* icon_bookmark                 = (char*)u8"\ue022";  // ICON_MS_BOOKMARK
[[maybe_unused]] static const char* icon_box                      = (char*)u8"\ue023";  // ICON_MS_INVENTORY_2
[[maybe_unused]] static const char* icon_briefcase                = (char*)u8"\ue024";  // ICON_MS_WORK
[[maybe_unused]] static const char* icon_british_pound            = (char*)u8"\ue025";  // ICON_MS_POUND
[[maybe_unused]] static const char* icon_browser                  = (char*)u8"\ue026";  // ICON_MS_WEB
[[maybe_unused]] static const char* icon_brush                    = (char*)u8"\ue027";  // ICON_MS_BRUSH
[[maybe_unused]] static const char* icon_bug                      = (char*)u8"\ue028";  // ICON_MS_BUG_REPORT
[[maybe_unused]] static const char* icon_bullhorn                 = (char*)u8"\ue029";  // ICON_MS_CAMPAIGN
[[maybe_unused]] static const char* icon_calculator               = (char*)u8"\ue02A";  // ICON_MS_CALCULATOR
[[maybe_unused]] static const char* icon_calendar                 = (char*)u8"\ue02B";  // ICON_MS_CALENDAR_TODAY
[[maybe_unused]] static const char* icon_camera_slr               = (char*)u8"\ue02C";  // ICON_MS_PHOTO_CAMERA
[[maybe_unused]] static const char* icon_caret_bottom             = (char*)u8"\ue02D";  // ICON_MS_ARROW_DROP_DOWN
[[maybe_unused]] static const char* icon_caret_left               = (char*)u8"\ue02E";  // ICON_MS_ARROW_LEFT
[[maybe_unused]] static const char* icon_caret_right              = (char*)u8"\ue02F";  // ICON_MS_ARROW_RIGHT
[[maybe_unused]] static const char* icon_caret_top                = (char*)u8"\ue030";  // ICON_MS_ARROW_DROP_UP
[[maybe_unused]] static const char* icon_cart                     = (char*)u8"\ue031";  // ICON_MS_SHOPPING_CART
[[maybe_unused]] static const char* icon_chat                     = (char*)u8"\ue032";  // ICON_MS_CHAT
[[maybe_unused]] static const char* icon_check                    = (char*)u8"\ue033";  // ICON_MS_CHECK
[[maybe_unused]] static const char* icon_chevron_bottom           = (char*)u8"\ue034";  // ICON_MS_KEYBOARD_ARROW_DOWN
[[maybe_unused]] static const char* icon_chevron_left             = (char*)u8"\ue035";  // ICON_MS_KEYBOARD_ARROW_LEFT
[[maybe_unused]] static const char* icon_chevron_right            = (char*)u8"\ue036";  // ICON_MS_KEYBOARD_ARROW_RIGHT
[[maybe_unused]] static const char* icon_chevron_top              = (char*)u8"\ue037";  // ICON_MS_KEYBOARD_ARROW_UP
[[maybe_unused]] static const char* icon_circle_check             = (char*)u8"\ue038";  // ICON_MS_CHECK_CIRCLE
[[maybe_unused]] static const char* icon_circle_x                 = (char*)u8"\ue039";  // ICON_MS_CANCEL
[[maybe_unused]] static const char* icon_clipboard                = (char*)u8"\ue03A";  // ICON_MS_CONTENT_PASTE
[[maybe_unused]] static const char* icon_clock                    = (char*)u8"\ue03B";  // ICON_MS_ACCESS_TIME
[[maybe_unused]] static const char* icon_cloud_download           = (char*)u8"\ue03C";  // ICON_MS_CLOUD_DOWNLOAD
[[maybe_unused]] static const char* icon_cloud_upload             = (char*)u8"\ue03D";  // ICON_MS_CLOUD_UPLOAD
[[maybe_unused]] static const char* icon_cloud                    = (char*)u8"\ue03E";  // ICON_MS_CLOUD
[[maybe_unused]] static const char* icon_cloudy                   = (char*)u8"\ue03F";  // ICON_MS_WB_CLOUDY
[[maybe_unused]] static const char* icon_code                     = (char*)u8"\ue040";  // ICON_MS_CODE
[[maybe_unused]] static const char* icon_cog                      = (char*)u8"\ue041";  // ICON_MS_SETTINGS
[[maybe_unused]] static const char* icon_collapse_down            = (char*)u8"\ue042";  // ICON_MS_UNFOLD_LESS
[[maybe_unused]] static const char* icon_collapse_left            = (char*)u8"\ue043";  // ICON_MS_ARROW_MENU_CLOSE
[[maybe_unused]] static const char* icon_collapse_right           = (char*)u8"\ue044";  // ICON_MS_ARROW_MENU_OPEN
[[maybe_unused]] static const char* icon_collapse_up              = (char*)u8"\ue045";  // ICON_MS_UNFOLD_MORE
[[maybe_unused]] static const char* icon_command                  = (char*)u8"\ue046";  // ICON_MS_KEYBOARD_COMMAND_KEY
[[maybe_unused]] static const char* icon_comment_square           = (char*)u8"\ue047";  // ICON_MS_COMMENT
[[maybe_unused]] static const char* icon_compass                  = (char*)u8"\ue048";  // ICON_MS_EXPLORE
[[maybe_unused]] static const char* icon_contrast                 = (char*)u8"\ue049";  // ICON_MS_CONTRAST
[[maybe_unused]] static const char* icon_copywriting              = (char*)u8"\ue04A";  // ICON_MS_CREATE
[[maybe_unused]] static const char* icon_credit_card              = (char*)u8"\ue04B";  // ICON_MS_CREDIT_CARD
[[maybe_unused]] static const char* icon_crop                     = (char*)u8"\ue04C";  // ICON_MS_CROP
[[maybe_unused]] static const char* icon_dashboard                = (char*)u8"\ue04D";  // ICON_MS_DASHBOARD
[[maybe_unused]] static const char* icon_data_transfer_download   = (char*)u8"\ue04E";  // ICON_MS_DOWNLOAD
[[maybe_unused]] static const char* icon_data_transfer_upload     = (char*)u8"\ue04F";  // ICON_MS_UPLOAD
[[maybe_unused]] static const char* icon_delete                   = (char*)u8"\ue050";  // ICON_MS_DELETE
[[maybe_unused]] static const char* icon_dial                     = (char*)u8"\ue051";  // ICON_MS_DIALPAD
[[maybe_unused]] static const char* icon_document                 = (char*)u8"\ue052";  // ICON_MS_DESCRIPTION
[[maybe_unused]] static const char* icon_dollar                   = (char*)u8"\ue053";  // ICON_MS_ATTACH_MONEY
[[maybe_unused]] static const char* icon_double_quote_sans_left   = (char*)u8"\ue054";  // ICON_MS_FORMAT_QUOTE
[[maybe_unused]] static const char* icon_double_quote_sans_right  = (char*)u8"\ue055";  // ICON_MS_FORMAT_QUOTE_CLOSE
[[maybe_unused]] static const char* icon_double_quote_serif_left  = (char*)u8"\ue056";  // ICON_MS_FORMAT_QUOTE
[[maybe_unused]] static const char* icon_double_quote_serif_right = (char*)u8"\ue057";  // ICON_MS_FORMAT_QUOTE_CLOSE
[[maybe_unused]] static const char* icon_droplet                  = (char*)u8"\ue058";  // ICON_MS_INVERT_COLORS
[[maybe_unused]] static const char* icon_eject                    = (char*)u8"\ue059";  // ICON_MS_EJECT
[[maybe_unused]] static const char* icon_elevator                 = (char*)u8"\ue05A";  // ICON_MS_ELEVATOR
[[maybe_unused]] static const char* icon_ellipses                 = (char*)u8"\ue05B";  // ICON_MS_MORE_HORIZ
[[maybe_unused]] static const char* icon_envelope_closed          = (char*)u8"\ue05C";  // ICON_MS_EMAIL
[[maybe_unused]] static const char* icon_envelope_open            = (char*)u8"\ue05D";  // ICON_MS_MARK_EMAIL_READ
[[maybe_unused]] static const char* icon_euro                     = (char*)u8"\ue05E";  // ICON_MS_EURO
[[maybe_unused]] static const char* icon_excerpt                  = (char*)u8"\ue05F";  // ICON_MS_SHORT_TEXT
[[maybe_unused]] static const char* icon_expend_down              = (char*)u8"\ue060";  // ICON_MS_UNFOLD_MORE
[[maybe_unused]] static const char* icon_expend_left              = (char*)u8"\ue061";  // ICON_MS_ARROW_MENU_OPEN
[[maybe_unused]] static const char* icon_expend_right             = (char*)u8"\ue062";  // ICON_MS_ARROW_MENU_CLOSE
[[maybe_unused]] static const char* icon_expend_up                = (char*)u8"\ue063";  // ICON_MS_UNFOLD_LESS
[[maybe_unused]] static const char* icon_external_link            = (char*)u8"\ue064";  // ICON_MS_OPEN_IN_NEW
[[maybe_unused]] static const char* icon_eye                      = (char*)u8"\ue065";  // ICON_MS_VISIBILITY
[[maybe_unused]] static const char* icon_eyedropper               = (char*)u8"\ue066";  // ICON_MS_COLORIZE
[[maybe_unused]] static const char* icon_file                     = (char*)u8"\ue067";  // ICON_MS_INSERT_DRIVE_FILE
[[maybe_unused]] static const char* icon_fire                     = (char*)u8"\ue068";  // ICON_MS_LOCAL_FIRE_DEPARTMENT
[[maybe_unused]] static const char* icon_flag                     = (char*)u8"\ue069";  // ICON_MS_FLAG
[[maybe_unused]] static const char* icon_flash                    = (char*)u8"\ue06A";  // ICON_MS_BOLT
[[maybe_unused]] static const char* icon_folder                   = (char*)u8"\ue06B";  // ICON_MS_FOLDER
[[maybe_unused]] static const char* icon_fork                     = (char*)u8"\ue06C";  // ICON_MS_CALL_SPLIT
[[maybe_unused]] static const char* icon_fullscreen_enter         = (char*)u8"\ue06D";  // ICON_MS_FULLSCREEN
[[maybe_unused]] static const char* icon_fullscreen_exit          = (char*)u8"\ue06E";  // ICON_MS_FULLSCREEN_EXIT
[[maybe_unused]] static const char* icon_globe                    = (char*)u8"\ue06F";  // ICON_MS_PUBLIC
[[maybe_unused]] static const char* icon_graph                    = (char*)u8"\ue070";  // ICON_MS_SHOW_CHART
[[maybe_unused]] static const char* icon_grid_four_up             = (char*)u8"\ue071";  // ICON_MS_VIEW_MODULE
[[maybe_unused]] static const char* icon_grid_three_up            = (char*)u8"\ue072";  // ICON_MS_VIEW_COMFY
[[maybe_unused]] static const char* icon_grid_two_up              = (char*)u8"\ue073";  // ICON_MS_VIEW_LIST
[[maybe_unused]] static const char* icon_hard_drive               = (char*)u8"\ue074";  // ICON_MS_STORAGE
[[maybe_unused]] static const char* icon_header                   = (char*)u8"\ue075";  // ICON_MS_TITLE
[[maybe_unused]] static const char* icon_headphones               = (char*)u8"\ue076";  // ICON_MS_HEADPHONES
[[maybe_unused]] static const char* icon_heart                    = (char*)u8"\ue077";  // ICON_MS_FAVORITE
[[maybe_unused]] static const char* icon_home                     = (char*)u8"\ue078";  // ICON_MS_HOME
[[maybe_unused]] static const char* icon_image                    = (char*)u8"\ue079";  // ICON_MS_IMAGE
[[maybe_unused]] static const char* icon_inbox                    = (char*)u8"\ue07A";  // ICON_MS_INBOX
[[maybe_unused]] static const char* icon_infinity                 = (char*)u8"\ue07B";  // ICON_MS_INFINITY
[[maybe_unused]] static const char* icon_info                     = (char*)u8"\ue07C";  // ICON_MS_INFO
[[maybe_unused]] static const char* icon_italic                   = (char*)u8"\ue07D";  // ICON_MS_ITALIC
[[maybe_unused]] static const char* icon_justify_center           = (char*)u8"\ue07E";  // ICON_MS_JUSTIFY_CENTER
[[maybe_unused]] static const char* icon_justify_left             = (char*)u8"\ue07F";  // ICON_MS_JUSTIFY_LEFT
[[maybe_unused]] static const char* icon_justify_right            = (char*)u8"\ue080";  // ICON_MS_JUSTIFY_RIGHT
[[maybe_unused]] static const char* icon_key                      = (char*)u8"\ue081";  // ICON_MS_VPN_KEY
[[maybe_unused]] static const char* icon_laptop                   = (char*)u8"\ue082";  // ICON_MS_LAPTOP
[[maybe_unused]] static const char* icon_layers                   = (char*)u8"\ue083";  // ICON_MS_LAYERS
[[maybe_unused]] static const char* icon_lightbulb                = (char*)u8"\ue084";  // ICON_MS_LIGHTBULB
[[maybe_unused]] static const char* icon_link_broken              = (char*)u8"\ue085";  // ICON_MS_LINK_OFF
[[maybe_unused]] static const char* icon_link_intact              = (char*)u8"\ue086";  // ICON_MS_LINK
[[maybe_unused]] static const char* icon_list                     = (char*)u8"\ue087";  // ICON_MS_LIST
[[maybe_unused]] static const char* icon_list_rich                = (char*)u8"\ue088";  // ICON_MS_VIEW_LIST
[[maybe_unused]] static const char* icon_location                 = (char*)u8"\ue089";  // ICON_MS_LOCATION_ON
[[maybe_unused]] static const char* icon_lock_locked              = (char*)u8"\ue08A";  // ICON_MS_LOCK
[[maybe_unused]] static const char* icon_lock_unlocked            = (char*)u8"\ue08B";  // ICON_MS_LOCK_OPEN
[[maybe_unused]] static const char* icon_loop_circular            = (char*)u8"\ue08C";  // ICON_MS_LOOP
[[maybe_unused]] static const char* icon_loop_square              = (char*)u8"\ue08D";  // ICON_MS_REPEAT
[[maybe_unused]] static const char* icon_loop                     = (char*)u8"\ue08E";  // ICON_MS_AUTORENEW
[[maybe_unused]] static const char* icon_magnifying_glass         = (char*)u8"\ue08F";  // ICON_MS_SEARCH
[[maybe_unused]] static const char* icon_map                      = (char*)u8"\ue090";  // ICON_MS_MAP
[[maybe_unused]] static const char* icon_map_marquer              = (char*)u8"\ue091";  // ICON_MS_ROOM
[[maybe_unused]] static const char* icon_media_pause              = (char*)u8"\ue092";  // ICON_MS_PAUSE
[[maybe_unused]] static const char* icon_media_play               = (char*)u8"\ue093";  // ICON_MS_PLAY_ARROW
[[maybe_unused]] static const char* icon_media_record             = (char*)u8"\ue094";  // ICON_MS_FIBER_MANUAL_RECORD
[[maybe_unused]] static const char* icon_media_skip_backward      = (char*)u8"\ue095";  // ICON_MS_SKIP_PREVIOUS
[[maybe_unused]] static const char* icon_media_skip_forward       = (char*)u8"\ue096";  // ICON_MS_SKIP_NEXT
[[maybe_unused]] static const char* icon_media_step_backward      = (char*)u8"\ue097";  // ICON_MS_FAST_REWIND
[[maybe_unused]] static const char* icon_media_step_forward       = (char*)u8"\ue098";  // ICON_MS_FAST_FORWARD
[[maybe_unused]] static const char* icon_media_stop               = (char*)u8"\ue099";  // ICON_MS_STOP
[[maybe_unused]] static const char* icon_medical_cross            = (char*)u8"\ue09A";  // ICON_MS_LOCAL_HOSPITAL
[[maybe_unused]] static const char* icon_menu                     = (char*)u8"\ue09B";  // ICON_MS_MENU
[[maybe_unused]] static const char* icon_microphone               = (char*)u8"\ue09C";  // ICON_MS_MIC
[[maybe_unused]] static const char* icon_minus                    = (char*)u8"\ue09D";  // ICON_MS_REMOVE
[[maybe_unused]] static const char* icon_monitor                  = (char*)u8"\ue09E";  // ICON_MS_DESKTOP_WINDOWS
[[maybe_unused]] static const char* icon_moon                     = (char*)u8"\ue09F";  // ICON_MS_BRIGHTNESS_2
[[maybe_unused]] static const char* icon_move                     = (char*)u8"\ue0A0";  // ICON_MS_OPEN_WITH
[[maybe_unused]] static const char* icon_musical_note             = (char*)u8"\ue0A1";  // ICON_MS_MUSIC_NOTE
[[maybe_unused]] static const char* icon_paperclip                = (char*)u8"\ue0A2";  // ICON_MS_ATTACH_FILE
[[maybe_unused]] static const char* icon_pencil                   = (char*)u8"\ue0A3";  // ICON_MS_EDIT
[[maybe_unused]] static const char* icon_people                   = (char*)u8"\ue0A4";  // ICON_MS_GROUP
[[maybe_unused]] static const char* icon_person                   = (char*)u8"\ue0A5";  // ICON_MS_PERSON
[[maybe_unused]] static const char* icon_phone                    = (char*)u8"\ue0A6";  // ICON_MS_PHONE
[[maybe_unused]] static const char* icon_pie_chart                = (char*)u8"\ue0A7";  // ICON_MS_PIE_CHART
[[maybe_unused]] static const char* icon_pin                      = (char*)u8"\ue0A8";  // ICON_MS_PUSH_PIN
[[maybe_unused]] static const char* icon_play_circle              = (char*)u8"\ue0A9";  // ICON_MS_PLAY_CIRCLE_FILLED
[[maybe_unused]] static const char* icon_plus                     = (char*)u8"\ue0AA";  // ICON_MS_ADD
[[maybe_unused]] static const char* icon_power_standby            = (char*)u8"\ue0AB";  // ICON_MS_POWER_SETTINGS_NEW
[[maybe_unused]] static const char* icon_print                    = (char*)u8"\ue0AC";  // ICON_MS_PRINT
[[maybe_unused]] static const char* icon_project                  = (char*)u8"\ue0AD";  // ICON_MS_INBOX
[[maybe_unused]] static const char* icon_pulse                    = (char*)u8"\ue0AE";  // ICON_MS_PULSE
[[maybe_unused]] static const char* icon_puzzle_piece             = (char*)u8"\ue0AF";  // ICON_MS_EXTENSION
[[maybe_unused]] static const char* icon_question_mark            = (char*)u8"\ue0B0";  // ICON_MS_HELP
[[maybe_unused]] static const char* icon_rain                     = (char*)u8"\ue0B1";  // ICON_MS_UMBRELLA
[[maybe_unused]] static const char* icon_random                   = (char*)u8"\ue0B2";  // ICON_MS_SHUFFLE
[[maybe_unused]] static const char* icon_reload                   = (char*)u8"\ue0B3";  // ICON_MS_REFRESH
[[maybe_unused]] static const char* icon_resize_both              = (char*)u8"\ue0B4";  // ICON_MS_OPEN_IN_FULL
[[maybe_unused]] static const char* icon_resize_height            = (char*)u8"\ue0B5";  // ICON_MS_VERTICAL_SPLIT
[[maybe_unused]] static const char* icon_resize_width             = (char*)u8"\ue0B6";  // ICON_MS_HORIZONTAL_SPLIT
[[maybe_unused]] static const char* icon_rss                      = (char*)u8"\ue0B7";  // ICON_MS_RSS_FEED
[[maybe_unused]] static const char* icon_rss_alt                  = (char*)u8"\ue0B8";  // ICON_MS_DYNAMIC_FEED
[[maybe_unused]] static const char* icon_script                   = (char*)u8"\ue0B9";  // ICON_MS_CODE
[[maybe_unused]] static const char* icon_share                    = (char*)u8"\ue0BA";  // ICON_MS_SHARE
[[maybe_unused]] static const char* icon_share_boxed              = (char*)u8"\ue0BB";  // ICON_MS_SHARE_WINDOW
[[maybe_unused]] static const char* icon_shield                   = (char*)u8"\ue0BC";  // ICON_MS_SECURITY
[[maybe_unused]] static const char* icon_signal                   = (char*)u8"\ue0BD";  // ICON_MS_SIGNAL_CELLULAR_4_BAR
[[maybe_unused]] static const char* icon_signpost                 = (char*)u8"\ue0BE";  // ICON_MS_SIGNPOST
[[maybe_unused]] static const char* icon_sort_ascending           = (char*)u8"\ue0BF";  // ICON_MS_SORT_BY_ALPHA
[[maybe_unused]] static const char* icon_sort_descending          = (char*)u8"\ue0C0";  // ICON_MS_SORT
[[maybe_unused]] static const char* icon_spreadsheet              = (char*)u8"\ue0C1";  // ICON_MS_TABLE_CHART
[[maybe_unused]] static const char* icon_star                     = (char*)u8"\ue0C2";  // ICON_MS_STAR
[[maybe_unused]] static const char* icon_sun                      = (char*)u8"\ue0C3";  // ICON_MS_SUNNY
[[maybe_unused]] static const char* icon_tablet                   = (char*)u8"\ue0C4";  // ICON_MS_TABLET
[[maybe_unused]] static const char* icon_tag                      = (char*)u8"\ue0C5";  // ICON_MS_LABEL
[[maybe_unused]] static const char* icon_tags                     = (char*)u8"\ue0C6";  // ICON_MS_LOCAL_OFFER
[[maybe_unused]] static const char* icon_target                   = (char*)u8"\ue0C7";  // ICON_MS_TARGETING
[[maybe_unused]] static const char* icon_task                     = (char*)u8"\ue0C8";  // ICON_MS_CHECKBOX
[[maybe_unused]] static const char* icon_terminal                 = (char*)u8"\ue0C9";  // ICON_MS_TERMINAL
[[maybe_unused]] static const char* icon_text                     = (char*)u8"\ue0CA";  // ICON_MS_TEXT_FIELDS
[[maybe_unused]] static const char* icon_thumb_down               = (char*)u8"\ue0CB";  // ICON_MS_THUMB_DOWN
[[maybe_unused]] static const char* icon_thumb_up                 = (char*)u8"\ue0CC";  // ICON_MS_THUMB_UP
[[maybe_unused]] static const char* icon_timer                    = (char*)u8"\ue0CD";  // ICON_MS_TIMER
[[maybe_unused]] static const char* icon_transfer                 = (char*)u8"\ue0CE";  // ICON_MS_COMPARE_ARROWS
[[maybe_unused]] static const char* icon_trash                    = (char*)u8"\ue0CF";  // ICON_MS_DELETE
[[maybe_unused]] static const char* icon_underline                = (char*)u8"\ue0D0";  // ICON_MS_UNDERLINE
[[maybe_unused]] static const char* icon_vertical_align_bottom    = (char*)u8"\ue0D1";  // ICON_MS_VERTICAL_ALIGN_BOTTOM
[[maybe_unused]] static const char* icon_vertical_align_center    = (char*)u8"\ue0D2";  // ICON_MS_VERTICAL_ALIGN_CENTER
[[maybe_unused]] static const char* icon_vertical_align_top       = (char*)u8"\ue0D3";  // ICON_MS_VERTICAL_ALIGN_TOP
[[maybe_unused]] static const char* icon_video                    = (char*)u8"\ue0D4";  // ICON_MS_VIDEOCAM
[[maybe_unused]] static const char* icon_volume_high              = (char*)u8"\ue0D5";  // ICON_MS_VOLUME_UP
[[maybe_unused]] static const char* icon_volume_low               = (char*)u8"\ue0D6";  // ICON_MS_VOLUME_DOWN
[[maybe_unused]] static const char* icon_volume_off               = (char*)u8"\ue0D7";  // ICON_MS_VOLUME_OFF
[[maybe_unused]] static const char* icon_warning                  = (char*)u8"\ue0D8";  // ICON_MS_WARNING
[[maybe_unused]] static const char* icon_wifi                     = (char*)u8"\ue0D9";  // ICON_MS_WIFI
[[maybe_unused]] static const char* icon_wrench                   = (char*)u8"\ue0DA";  // ICON_MS_BUILD
[[maybe_unused]] static const char* icon_x                        = (char*)u8"\ue0DB";  // ICON_MS_CLOSE
[[maybe_unused]] static const char* icon_yen                      = (char*)u8"\ue0DC";  // ICON_MS_YEN
[[maybe_unused]] static const char* icon_zoom_in                  = (char*)u8"\ue0DD";  // ICON_MS_ZOOM_IN
[[maybe_unused]] static const char* icon_zoom_out                 = (char*)u8"\ue0DE";  // ICON_MS_ZOOM_OUT

}  // namespace nvgui
