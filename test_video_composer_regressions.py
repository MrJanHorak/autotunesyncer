#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_PATH = os.path.join(os.path.dirname(__file__), 'backend')
BACKEND_PYTHON_PATH = os.path.join(BACKEND_PATH, 'python')
for path in (BACKEND_PATH, BACKEND_PYTHON_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

import python.video_composer as video_composer_module
from python.video_composer import VideoComposer


class VideoComposerRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_ctx.name)
        self.addCleanup(self.temp_dir_ctx.cleanup)

    def make_composer(self, **overrides):
        composer = VideoComposer.__new__(VideoComposer)
        composer.temp_dir = self.temp_dir
        composer.processed_videos_dir = self.temp_dir / 'processed'
        composer.processed_videos_dir.mkdir(exist_ok=True)
        composer.uploads_dir = self.temp_dir / 'uploads'
        composer.uploads_dir.mkdir(exist_ok=True)
        composer.composition_style = {'backgroundColor': '#112233'}
        composer.clip_styles = {}
        composer.grid_positions = {}
        composer.render_config = {
            'resolution': '1920x1080',
            'preset': 'fast',
            'crf': '26',
            'audio_bitrate': '192k',
            'video_bitrate': '3M',
            'scale_filter': 'scale=1920:1080',
        }
        composer.preview_mode = False
        composer.max_concurrent_streams = 32
        composer.ffmpeg_hwaccel = None

        for key, value in overrides.items():
            setattr(composer, key, value)
        return composer

    def write_dummy_media(self, name='clip.mp4'):
        media_path = self.temp_dir / name
        media_path.write_bytes(b'not-a-real-mp4')
        return media_path

    def test_resolve_clip_style_matches_prefixed_frontend_keys(self):
        composer = self.make_composer(
            clip_styles={
                'track-2': {'bgColor': '#123456'},
                'drum-drum_snare_drum': {'labelText': 'Snare'},
            }
        )

        instrument_style, instrument_candidates, instrument_key = composer._resolve_clip_style('2')
        drum_style, drum_candidates, drum_key = composer._resolve_clip_style('drum_snare_drum')

        self.assertEqual(instrument_key, 'track-2')
        self.assertEqual(instrument_style['bgColor'], '#123456')
        self.assertIn('track-2', instrument_candidates)
        self.assertEqual(drum_key, 'drum-drum_snare_drum')
        self.assertEqual(drum_style['labelText'], 'Snare')
        self.assertIn('drum-drum_snare_drum', drum_candidates)

    def test_note_triggered_sequence_uses_clip_background_color(self):
        source_path = self.write_dummy_media('snare-source.mp4')
        composer = self.make_composer(
            composition_style={'backgroundColor': '#0000AA'},
            clip_styles={
                'drum-drum_snare_drum': {
                    'bgColorEnabled': True,
                    'bgColor': '#1cb52b',
                    'transparentBg': False,
                }
            },
        )
        captured = {}

        composer._get_media_duration = lambda _path: 8.0
        composer._build_atempo_chain = lambda _value: 'atempo=1.0'
        composer._create_simple_loop = lambda *_args, **_kwargs: 'loop-fallback.mp4'

        def fake_run_ffmpeg(cmd, filter_parts, tail_args):
            captured['cmd'] = cmd
            captured['filter_parts'] = filter_parts
            captured['tail_args'] = tail_args
            return SimpleNamespace(returncode=0, stderr='')

        composer._run_ffmpeg_with_filter_script = fake_run_ffmpeg

        result = composer._create_note_triggered_video_sequence_fixed(
            str(source_path),
            [{'time': 0.0, 'duration': 0.4, 'midi': 60}],
            1.0,
            'snare',
            'regression',
            style_track_id='drum_snare_drum',
        )

        self.assertEqual(result, str(self.temp_dir / 'snare_regression.mp4'))
        self.assertIn(
            'color=0x1CB52B:size=640x360:rate=30:duration=1.0',
            captured['cmd'],
        )

    def test_background_video_keeps_note_trigger_output_transparent(self):
        source_path = self.write_dummy_media('lead-source.mp4')
        composer = self.make_composer(
            background_media={'kind': 'video', 'path': str(self.write_dummy_media('stage-bg.mp4'))},
            composition_style={'backgroundColor': '#112233', 'backgroundMode': 'video'},
            clip_styles={'track-4': {'transparentBg': False}},
        )
        captured = {}

        composer._get_media_duration = lambda _path: 8.0
        composer._build_atempo_chain = lambda _value: 'atempo=1.0'
        composer._create_simple_loop = lambda *_args, **_kwargs: 'loop-fallback.mp4'

        def fake_run_ffmpeg(cmd, filter_parts, tail_args):
            captured['cmd'] = cmd
            captured['filter_parts'] = filter_parts
            captured['tail_args'] = tail_args
            return SimpleNamespace(returncode=0, stderr='')

        composer._run_ffmpeg_with_filter_script = fake_run_ffmpeg

        result = composer._create_note_triggered_video_sequence_fixed(
            str(source_path),
            [{'time': 0.0, 'duration': 0.4, 'midi': 60}],
            1.0,
            'lead',
            'with-bg',
            style_track_id='4',
        )

        self.assertEqual(result, str(self.temp_dir / 'lead_with-bg.mov'))
        self.assertIn(
            'color=c=black@0.0:size=640x360:rate=30:duration=1.0',
            captured['cmd'],
        )
        alpha_filter = ''.join(captured['filter_parts'])
        self.assertIn('format=rgba', alpha_filter)
        self.assertIn('tpad=stop_mode=clone:stop_duration=1.000', alpha_filter)

    def test_transparent_note_trigger_output_clones_last_visible_frame(self):
        source_path = self.write_dummy_media('alpha-source.mov')
        composer = self.make_composer(
            clip_styles={'track-2': {'transparentBg': True}},
        )
        captured = {}

        composer._get_media_duration = lambda _path: 0.6
        composer._build_atempo_chain = lambda _value: 'atempo=1.0'
        composer._create_simple_loop = lambda *_args, **_kwargs: 'loop-fallback.mov'

        def fake_run_ffmpeg(cmd, filter_parts, tail_args):
            captured['cmd'] = cmd
            captured['filter_parts'] = filter_parts
            captured['tail_args'] = tail_args
            return SimpleNamespace(returncode=0, stderr='')

        composer._run_ffmpeg_with_filter_script = fake_run_ffmpeg

        result = composer._create_note_triggered_video_sequence_fixed(
            str(source_path),
            [{'time': 0.0, 'duration': 1.0, 'midi': 60}],
            1.2,
            'alpha',
            'clone-last-frame',
            style_track_id='2',
        )

        self.assertEqual(result, str(self.temp_dir / 'alpha_clone-last-frame.mov'))
        alpha_filter = ''.join(captured['filter_parts'])
        self.assertIn('format=rgba', alpha_filter)
        self.assertIn('tpad=stop_mode=clone:stop_duration=1.200', alpha_filter)
        self.assertIn('trim=duration=1.200,setpts=PTS-STARTPTS+0.000/TB[v0]', alpha_filter)

    def test_note_trigger_video_segments_are_shifted_to_note_time(self):
        source_path = self.write_dummy_media('retrigger-source.mp4')
        composer = self.make_composer()
        captured = {}

        composer._get_media_duration = lambda _path: 8.0
        composer._build_atempo_chain = lambda _value: 'atempo=1.0'
        composer._create_simple_loop = lambda *_args, **_kwargs: 'loop-fallback.mp4'

        def fake_run_ffmpeg(cmd, filter_parts, tail_args):
            captured['cmd'] = cmd
            captured['filter_parts'] = filter_parts
            captured['tail_args'] = tail_args
            return SimpleNamespace(returncode=0, stderr='')

        composer._run_ffmpeg_with_filter_script = fake_run_ffmpeg

        result = composer._create_note_triggered_video_sequence_fixed(
            str(source_path),
            [
                {'time': 0.0, 'duration': 0.4, 'midi': 60},
                {'time': 2.0, 'duration': 0.4, 'midi': 62},
            ],
            3.0,
            'retrigger',
            'timed-notes',
        )

        self.assertEqual(result, str(self.temp_dir / 'retrigger_timed-notes.mp4'))
        filter_graph = ''.join(captured['filter_parts'])
        self.assertIn('setpts=PTS-STARTPTS+2.000/TB[v1]', filter_graph)
        self.assertIn("enable='between(t,2.000,3.000)'", filter_graph)

    def test_cell_style_filters_use_lanczos_scaling(self):
        composer = self.make_composer()
        filter_parts = []

        composer._apply_cell_style_filters(
            filter_parts,
            '[0:v]',
            '[v0]',
            640,
            360,
            'track-0',
            {'notes': []},
            [],
            chunk_duration=1.0,
            beat_sync_stats={},
        )

        self.assertTrue(filter_parts)
        self.assertIn(
            'scale=640:360:flags=lanczos:force_original_aspect_ratio=increase',
            filter_parts[0],
        )

    def test_cell_style_border_is_gated_to_active_note_windows(self):
        composer = self.make_composer(
            clip_styles={
                'track-0': {
                    'borderWidth': 3,
                    'borderColor': '#ff44aa',
                }
            }
        )
        filter_parts = []

        composer._apply_cell_style_filters(
            filter_parts,
            '[0:v]',
            '[v0]',
            640,
            360,
            '0',
            {
                'notes': [
                    {'chunk_time': 0.25, 'duration': 0.5},
                ]
            },
            [],
            chunk_duration=1.0,
            beat_sync_stats={},
        )

        border_filters = [
            part for part in filter_parts if 'drawbox=' in part and ':t=3' in part
        ]

        self.assertEqual(len(border_filters), 1)
        self.assertIn(
            ":enable='between(t,0.25,0.75)'",
            border_filters[0],
        )

    def test_cell_style_rounded_corners_replace_alpha_pixels_for_stage_background(self):
        composer = self.make_composer(
            composition_style={'backgroundColor': '#112233', 'backgroundMode': 'video'},
            clip_styles={
                'track-0': {
                    'roundedCorners': True,
                    'cornerRadius': 18,
                }
            },
        )
        filter_parts = []

        composer._apply_cell_style_filters(
            filter_parts,
            '[0:v]',
            '[v0]',
            640,
            360,
            '0',
            {'notes': [], 'preserve_idle_alpha': True},
            [],
            chunk_duration=1.0,
            beat_sync_stats={},
        )

        rounded_filter = ''.join(filter_parts)
        self.assertIn('color=black@0.0:t=fill:replace=1', rounded_filter)

    def test_cell_style_rounded_corners_use_clip_background_color(self):
        composer = self.make_composer(
            composition_style={'backgroundColor': '#112233'},
            clip_styles={
                'track-0': {
                    'roundedCorners': True,
                    'cornerRadius': 18,
                    'bgColorEnabled': True,
                    'bgColor': '#1cb52b',
                }
            },
        )
        filter_parts = []

        composer._apply_cell_style_filters(
            filter_parts,
            '[0:v]',
            '[v0]',
            640,
            360,
            '0',
            {'notes': []},
            [],
            chunk_duration=1.0,
            beat_sync_stats={},
        )

        rounded_filter = ''.join(filter_parts)
        self.assertIn('color=0x1CB52B@1:t=fill', rounded_filter)

    def test_grid_layout_aborts_when_segment_has_no_mapping(self):
        source_path = self.write_dummy_media('grid-segment.mp4')
        composer = self.make_composer(grid_positions={'0': {'row': 0, 'column': 0}})
        composer._create_ffmpeg_grid_layout_fixed = Mock(return_value='unexpected.mp4')

        result = composer._create_grid_layout_chunk_fixed(
            [
                {
                    'track_id': 'missing-track',
                    'track_name': 'Missing Track',
                    'type': 'instrument',
                    'video_path': str(source_path),
                }
            ],
            self.temp_dir / 'grid-output.mp4',
            4.0,
        )

        self.assertIsNone(result)
        composer._create_ffmpeg_grid_layout_fixed.assert_not_called()

    def test_setup_track_configuration_preserves_v2_stage_and_spans(self):
        composer = self.make_composer(
            midi_data={
                'tracks': [
                    {
                        'channel': 0,
                        'instrument': {'name': 'Piano', 'family': 'piano'},
                        'notes': [{'midi': 60, 'time': 0.0, 'duration': 0.5}],
                    }
                ],
                'gridArrangement': {
                    'version': 2,
                    'columns': 12,
                    'rows': 12,
                    'items': {
                        'track_0_piano': {'x': 2, 'y': 3, 'w': 4, 'h': 5},
                    },
                },
            },
        )
        composer._analyze_midi_timing = lambda: None

        composer._setup_track_configuration()

        self.assertEqual(composer.grid_stage_cols, 12)
        self.assertEqual(composer.grid_stage_rows, 12)
        self.assertEqual(composer.grid_positions['track_0_piano']['column'], 2)
        self.assertEqual(composer.grid_positions['track_0_piano']['row'], 3)
        self.assertEqual(composer.grid_positions['track_0_piano']['w'], 4)
        self.assertEqual(composer.grid_positions['track_0_piano']['h'], 5)

    def test_placeholder_chunk_uses_composition_background(self):
        composer = self.make_composer(composition_style={'backgroundColor': '#224466'})
        captured = {}

        def fake_gpu_subprocess_run(cmd, capture_output=True, text=True):
            captured['cmd'] = cmd
            return SimpleNamespace(returncode=0, stderr='')

        with patch.object(video_composer_module, 'gpu_subprocess_run', side_effect=fake_gpu_subprocess_run):
            result = composer._create_placeholder_chunk_simple(7, self.temp_dir, 2.5)

        self.assertEqual(result, str(self.temp_dir / 'placeholder_chunk_7.mp4'))
        self.assertIn(
            'color=0x224466:size=1920x1080:duration=2.5:rate=30',
            captured['cmd'],
        )

    def test_single_active_cell_preserves_grid_when_solo_mode_is_disabled(self):
        source_path = self.write_dummy_media('solo-cell.mp4')
        composer = self.make_composer(
            composition_style={'backgroundColor': '#0F172A'},
            grid_positions={
                '0': {'row': 0, 'column': 0},
                'blank-1': {'row': 0, 'column': 1},
                'blank-2': {'row': 1, 'column': 0},
                'blank-3': {'row': 1, 'column': 1},
            },
        )
        captured = {}

        composer._preprocess_extend_clip = lambda clip_path, _duration, _bg_hex: (clip_path, False)
        composer._get_ffmpeg_decode_args = lambda: []
        composer._resolve_segment_volume = lambda _segment: 0.0
        composer._velocity_to_db = lambda _velocity: 0.0
        composer._get_encoding_settings = lambda: ['-c:v', 'libx264', '-preset', 'fast', '-crf', '26']

        def fake_apply_cell_style_filters(filter_parts, input_label, output_label, *_args, **_kwargs):
            filter_parts.append(f'{input_label}null{output_label}')

        composer._apply_cell_style_filters = fake_apply_cell_style_filters
        composer._apply_global_style_filters = (
            lambda _filter_parts, video_label, _width, _height, _duration, audio_label, _temp_files:
            (video_label, audio_label)
        )

        def fake_subprocess_run(cmd, capture_output=True, text=True):
            captured['cmd'] = cmd
            if '-filter_complex_script' in cmd:
                script_path = Path(cmd[cmd.index('-filter_complex_script') + 1])
                captured['filter_script'] = script_path.read_text(encoding='utf-8')
            elif '-filter_complex' in cmd:
                captured['filter_script'] = cmd[cmd.index('-filter_complex') + 1]
            return SimpleNamespace(returncode=0, stderr='', stdout='')

        with patch.dict(os.environ, {'ATS_ENABLE_SOLO_MODE': ''}, clear=False):
            with patch.object(video_composer_module.subprocess, 'run', side_effect=fake_subprocess_run):
                result = composer._create_ffmpeg_grid_layout_fixed(
                    [
                        {
                            'track_id': '0',
                            'track_name': 'Lead',
                            'type': 'instrument',
                            'video_path': str(source_path),
                        }
                    ],
                    self.temp_dir / 'single-cell-grid.mp4',
                    4.0,
                )

        self.assertEqual(result, str(self.temp_dir / 'single-cell-grid.mp4'))
        self.assertIn('filter_script', captured)
        self.assertIn('overlay=x=0:y=0:eof_action=pass:format=auto', captured['filter_script'])

    def test_span_aware_grid_layout_uses_stage_coordinates(self):
        source_path = self.write_dummy_media('span-cell.mp4')
        composer = self.make_composer(
            composition_style={'backgroundColor': '#0F172A'},
            grid_stage_rows=12,
            grid_stage_cols=12,
            grid_positions={
                '0': {'row': 2, 'column': 1, 'w': 4, 'h': 3},
            },
        )
        captured = {'cell_sizes': []}

        composer._preprocess_extend_clip = lambda clip_path, _duration, _bg_hex: (clip_path, False)
        composer._get_ffmpeg_decode_args = lambda: []
        composer._resolve_segment_volume = lambda _segment: 0.0
        composer._velocity_to_db = lambda _velocity: 0.0
        composer._get_encoding_settings = lambda: ['-c:v', 'libx264', '-preset', 'fast', '-crf', '26']

        def fake_apply_cell_style_filters(
            filter_parts,
            input_label,
            output_label,
            cell_w,
            cell_h,
            track_id,
            *_args,
            **_kwargs,
        ):
            captured['cell_sizes'].append((track_id, cell_w, cell_h))
            filter_parts.append(f'{input_label}null{output_label}')

        composer._apply_cell_style_filters = fake_apply_cell_style_filters
        composer._apply_global_style_filters = (
            lambda _filter_parts, video_label, _width, _height, _duration, audio_label, _temp_files:
            (video_label, audio_label)
        )

        def fake_subprocess_run(cmd, capture_output=True, text=True):
            captured['cmd'] = cmd
            if '-filter_complex_script' in cmd:
                script_path = Path(cmd[cmd.index('-filter_complex_script') + 1])
                captured['filter_script'] = script_path.read_text(encoding='utf-8')
            elif '-filter_complex' in cmd:
                captured['filter_script'] = cmd[cmd.index('-filter_complex') + 1]
            return SimpleNamespace(returncode=0, stderr='', stdout='')

        with patch.dict(os.environ, {'ATS_ENABLE_SOLO_MODE': ''}, clear=False):
            with patch.object(video_composer_module.subprocess, 'run', side_effect=fake_subprocess_run):
                result = composer._create_ffmpeg_grid_layout_fixed(
                    [
                        {
                            'track_id': '0',
                            'track_name': 'Lead',
                            'type': 'instrument',
                            'video_path': str(source_path),
                        }
                    ],
                    self.temp_dir / 'span-grid.mp4',
                    4.0,
                )

        self.assertEqual(result, str(self.temp_dir / 'span-grid.mp4'))
        self.assertEqual(captured['cell_sizes'], [('0', 640, 270)])
        self.assertIn('color=0x0F172A:s=1920x1080:r=30:d=4.0', captured['cmd'])
        self.assertIn('overlay=x=160:y=180:eof_action=pass:format=auto', captured['filter_script'])

    def test_legacy_midi_sync_wrapper_routes_to_create_composition(self):
        recorded = {}

        class DummyComposer(VideoComposer):
            def __init__(self, midi_data, processed_videos_dir, output_path, preview_mode=False):
                recorded['midi_data'] = midi_data
                recorded['processed_videos_dir'] = processed_videos_dir
                recorded['output_path'] = output_path
                recorded['preview_mode'] = preview_mode

            def create_composition(self):
                return 'compat-output.mp4'

        composer = DummyComposer.__new__(DummyComposer)
        composer.temp_dir = self.temp_dir
        composer.processed_videos_dir = self.temp_dir / 'processed'
        composer.uploads_dir = self.temp_dir / 'uploads'
        composer.preview_mode = True

        result = composer.create_midi_synchronized_composition(
            {'tracks': []},
            {'track-0': 'clip.mp4'},
            'final-output.mp4',
        )

        self.assertEqual(result, 'compat-output.mp4')
        self.assertEqual(recorded['midi_data']['videoFiles'], {'track-0': 'clip.mp4'})
        self.assertEqual(recorded['midi_data']['uploadsDir'], str(composer.uploads_dir))
        self.assertEqual(recorded['processed_videos_dir'], str(composer.processed_videos_dir))
        self.assertEqual(recorded['output_path'], 'final-output.mp4')
        self.assertTrue(recorded['preview_mode'])


if __name__ == '__main__':
    unittest.main(verbosity=2)