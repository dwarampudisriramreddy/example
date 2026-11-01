// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

const int _inputWidth = 640;
const int _inputHeight = 640;
const String _modelAsset = 'assets/models/conditions_float32.onnx';
const String _inputName = 'images';
const String _outputName = 'Identity';
const int _maxDetections = 300;
const double _defaultConfThreshold = 0.25;

const List<String> CLASS_NAMES = [
  "Normal",
  "Initial Caries",
  "Moderate Caries",
  "Severe Caries",
  "Tooth Stain",
  "Dental Calculus",
  "Other Lesions"
];

const Map<String, String> CLASS_DESCRIPTIONS = {
  "Normal": "Healthy tooth structure, no visible pathology.",
  "Initial Caries": "Early-stage enamel demineralization (white spot lesion).",
  "Moderate Caries": "Dentin involvement with a visible cavitated lesion.",
  "Severe Caries": "Extensive decay approaching or involving the pulp.",
  "Tooth Stain": "Extrinsic or intrinsic discoloration of the enamel.",
  "Dental Calculus": "Calcified plaque (tartar) deposits on the tooth surface.",
  "Other Lesions": "Other soft or hard tissue anomalies not listed above.",
};

final List<Color> _classColors = [
  Colors.teal,
  Colors.blue,
  Colors.orange,
  Colors.red,
  Colors.purple,
  Colors.amber,
  Colors.pink,
];

class _Detection {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
  final int classId;

  _Detection(this.x1, this.y1, this.x2, this.y2, this.score, this.classId);

  String get label => classId >= 0 && classId < CLASS_NAMES.length
      ? CLASS_NAMES[classId]
      : 'Class $classId';

  String get description =>
      CLASS_DESCRIPTIONS[label] ?? 'No description available.';
}

void main() {
  runApp(const ToothDetectionApp());
}

class ToothDetectionApp extends StatelessWidget {
  const ToothDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tooth Condition Detection',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.teal,
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
        ),
      ),
      home: const DetectionHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DetectionHomePage extends StatefulWidget {
  const DetectionHomePage({super.key});

  @override
  State<DetectionHomePage> createState() => _DetectionHomePageState();
}

class _DetectionHomePageState extends State<DetectionHomePage> {
  final OnnxRuntime _ort = OnnxRuntime();
  OrtSession? _session;
  bool _sessionReady = false;
  bool _isLoading = false;
  File? _imageFile;
  int _imageWidth = 0;
  int _imageHeight = 0;
  List<_Detection> _detections = [];
  double _confThreshold = _defaultConfThreshold;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializeSession();
  }

  @override
  void dispose() {
    _session?.close();
    super.dispose();
  }

  Future<void> _initializeSession() async {
    setState(() {
      _sessionReady = false;
      _error = null;
    });
    try {
      final options = OrtSessionOptions(
        providers: kIsWeb ? [OrtProvider.WEB_ASSEMBLY] : null,
      );
      _session = await _ort.createSessionFromAsset(_modelAsset, options: options);
      final inputInfo = await _session!.getInputInfo();
      final outputInfo = await _session!.getOutputInfo();
      debugPrint('Model inputs: $inputInfo');
      debugPrint('Model outputs: $outputInfo');
      if (mounted) {
        setState(() => _sessionReady = true);
      }
    } catch (e) {
      final detail = e is PlatformException
          ? 'code: ${e.code}, message: ${e.message}'
          : e.toString();
      if (mounted) {
        setState(() => _error = 'Failed to load model: $detail\n\nNote: on web, some ONNX operators are not supported by onnxruntime-web. Try a model converted with opset compatible with the web backend.');
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1600,
      maxHeight: 1600,
    );
    if (picked == null) return;
    await _processImage(File(picked.path));
  }

  Future<void> _processImage(File file) async {
    setState(() {
      _isLoading = true;
      _error = null;
      _detections = [];
      _imageFile = file;
    });

    try {
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Unable to decode image');
      _imageWidth = decoded.width;
      _imageHeight = decoded.height;

      if (!_sessionReady) await _initializeSession();
      if (_session == null) throw Exception('ONNX session is not ready');

      final inputTensor = await OrtValue.fromList(_preprocess(bytes), [1, _inputWidth, _inputHeight, 3]);
      final outputs = await _session!.run({_inputName: inputTensor});
      final outputTensor = outputs[_outputName];
      if (outputTensor == null) throw Exception('Model output "$_outputName" not found');

      final flat = await outputTensor.asFlattenedList();
      final detections = _postprocess(flat);

      for (final tensor in [inputTensor, outputTensor]) {
        await tensor.dispose();
      }

      if (mounted) {
        setState(() {
          _detections = detections;
          _isLoading = false;
        });
      }
    } catch (e) {
      final detail = e is PlatformException
          ? 'code: ${e.code}, message: ${e.message}'
          : e.toString();
      if (mounted) {
        setState(() {
          _error = 'Detection failed: $detail';
          _isLoading = false;
        });
      }
    }
  }

  Float32List _preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image for preprocessing');
    final resized = img.copyResize(image, width: _inputWidth, height: _inputHeight);
    final data = Float32List(1 * _inputWidth * _inputHeight * 3);
    var idx = 0;
    for (var y = 0; y < _inputHeight; y++) {
      for (var x = 0; x < _inputWidth; x++) {
        final p = resized.getPixel(x, y);
        data[idx++] = p.r / 255.0;
        data[idx++] = p.g / 255.0;
        data[idx++] = p.b / 255.0;
      }
    }
    return data;
  }

  List<_Detection> _postprocess(List<dynamic> flat) {
    final detections = <_Detection>[];
    for (var i = 0; i < _maxDetections; i++) {
      final base = i * 6;
      final x1 = (flat[base] as num).toDouble();
      final y1 = (flat[base + 1] as num).toDouble();
      final x2 = (flat[base + 2] as num).toDouble();
      final y2 = (flat[base + 3] as num).toDouble();
      final score = (flat[base + 4] as num).toDouble();
      final classId = ((flat[base + 5] as num).toDouble()).round();

      if (score <= 0 || score <= _confThreshold) continue;

      var nx1 = x1;
      var ny1 = y1;
      var nx2 = x2;
      var ny2 = y2;
      // Output may be in pixel coordinates relative to the 640x640 input.
      if (x1 > 1 || y1 > 1 || x2 > 1 || y2 > 1) {
        nx1 /= _inputWidth;
        ny1 /= _inputHeight;
        nx2 /= _inputWidth;
        ny2 /= _inputHeight;
      }
      nx1 = nx1.clamp(0.0, 1.0);
      ny1 = ny1.clamp(0.0, 1.0);
      nx2 = nx2.clamp(0.0, 1.0);
      ny2 = ny2.clamp(0.0, 1.0);
      if (nx2 <= nx1 || ny2 <= ny1) continue;

      detections.add(_Detection(nx1, ny1, nx2, ny2, score, classId));
    }
    detections.sort((a, b) => b.score.compareTo(a.score));
    return detections;
  }

  Widget _buildImageCanvas(BuildContext context, double canvasSide) {
    if (_imageFile == null || _imageWidth == 0 || _imageHeight == 0) {
      return Container(
        width: canvasSide,
        height: canvasSide,
        color: Colors.grey.shade100,
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.image_outlined, size: 48, color: Colors.grey),
              SizedBox(height: 8),
              Text('No image selected', style: TextStyle(color: Colors.grey)),
            ],
          ),
        ),
      );
    }

    final scale = (canvasSide / _imageWidth).clamp(0, double.infinity);
    final clampedScale = (canvasSide / _imageHeight).clamp(0, double.infinity);
    final fitScale = scale < clampedScale ? scale : clampedScale;
    final dispW = (_imageWidth * fitScale).toDouble();
    final dispH = (_imageHeight * fitScale).toDouble();
    final offsetX = ((canvasSide - dispW) / 2).toDouble();
    final offsetY = ((canvasSide - dispH) / 2).toDouble();

    return Stack(
      children: [
        Container(
          width: canvasSide,
          height: canvasSide,
          color: Colors.grey.shade100,
          alignment: Alignment.center,
          child: Image.file(
            _imageFile!,
            width: dispW,
            height: dispH,
            fit: BoxFit.fill,
          ),
        ),
        ..._detections.map((det) {
          final color = _classColors[det.classId % _classColors.length];
          final left = (offsetX + det.x1 * dispW).clamp(0.0, canvasSide - 1.0).toDouble();
          final top = (offsetY + det.y1 * dispH).clamp(0.0, canvasSide - 1.0).toDouble();
          final width = ((det.x2 - det.x1) * dispW).clamp(0.0, canvasSide - left).toDouble();
          final height = ((det.y2 - det.y1) * dispH).clamp(0.0, canvasSide - top).toDouble();
          return Positioned(
            left: left,
            top: top,
            child: SizedBox(
              width: width,
              height: height,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: color, width: 3),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  Positioned(
                    left: 0,
                    top: -20,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: color,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        '${det.label} ${(det.score * 100).toStringAsFixed(0)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final canvasSide = (size.width < size.height ? size.width : size.height) - 32;
    final clampedCanvas = canvasSide.clamp(200.0, 640.0);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Tooth Condition Detection'),
        actions: [
          if (_sessionReady)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _initializeSession,
              tooltip: 'Reload model',
            ),
        ],
      ),
      body: Column(
        children: [
          Card(
            margin: const EdgeInsets.all(12),
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Text('Confidence: '),
                      Expanded(
                        child: Slider(
                          value: _confThreshold,
                          min: 0.0,
                          max: 1.0,
                          divisions: 100,
                          label: (_confThreshold * 100).round().toString(),
                          onChanged: _isLoading
                              ? null
                              : (value) => setState(() => _confThreshold = value),
                        ),
                      ),
                      Text('${(_confThreshold * 100).round()}%'),
                    ],
                  ),
                  if (_error != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: Text(_error!, style: const TextStyle(color: Colors.red)),
                    ),
                  if (_imageFile != null && !_isLoading)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: Row(
                        children: [
                          const Icon(Icons.sync, size: 18),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              'Detected ${_detections.length} tooth condition(s)',
                              style: const TextStyle(fontWeight: FontWeight.w600),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
          Expanded(
            child: Center(
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : _buildImageCanvas(context, clampedCanvas),
            ),
          ),
          if (_detections.isNotEmpty)
            Expanded(
              flex: 2,
              child: ListView.builder(
                itemCount: _detections.length,
                itemBuilder: (context, i) {
                  final det = _detections[i];
                  final color = _classColors[det.classId % _classColors.length];
                  return ListTile(
                    leading: CircleAvatar(
                      backgroundColor: color.withValues(alpha: 0.2),
                      child: Text(
                        '${i + 1}',
                        style: TextStyle(color: color, fontWeight: FontWeight.bold),
                      ),
                    ),
                    title: Text(
                      det.label,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    subtitle: Text(
                      '${det.description}\nConfidence: ${(det.score * 100).toStringAsFixed(1)}%  •  Box: [${(det.x1 * 100).round()}, ${(det.y1 * 100).round()}, ${(det.x2 * 100).round()}, ${(det.y2 * 100).round()}]',
                    ),
                    trailing: Text(
                      '${(det.score * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  );
                },
              ),
            ),
        ],
      ),
      floatingActionButton: _sessionReady && _imageFile != null && !_isLoading
          ? FloatingActionButton.extended(
              icon: const Icon(Icons.search),
              label: const Text('Detect'),
              onPressed: () => _processImage(_imageFile!),
            )
          : null,
    );
  }
}
