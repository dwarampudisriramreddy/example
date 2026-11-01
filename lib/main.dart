// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

const int inputWidth = 640;
const int inputHeight = 640;
const String modelAsset = 'assets/models/conditions_float32.onnx';
const String inputName = 'images';
const String outputName = 'Identity';
const int maxDetections = 300;
const double defaultConfThreshold = 0.25;

List<CameraDescription> cameras = [];

const List<String> kClassNames = [
  "Normal",
  "Initial Caries",
  "Moderate Caries",
  "Severe Caries",
  "Tooth Stain",
  "Dental Calculus",
  "Other Lesions"
];

const Map<String, String> kClassDescriptions = {
  "Normal": "Healthy tooth structure, no visible pathology.",
  "Initial Caries": "Early-stage enamel demineralization (white spot lesion).",
  "Moderate Caries": "Dentin involvement with a visible cavitated lesion.",
  "Severe Caries": "Extensive decay approaching or involving the pulp.",
  "Tooth Stain": "Extrinsic or intrinsic discoloration of the enamel.",
  "Dental Calculus": "Calcified plaque (tartar) deposits on the tooth surface.",
  "Other Lesions": "Other soft or hard tissue anomalies not listed above.",
};

final List<Color> classColors = [
  Colors.green,
  Colors.red,
  Colors.red,
  Colors.red,
  Colors.red,
  Colors.red,
  Colors.red,
];

class Detection {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
  final int classId;

  Detection(this.x1, this.y1, this.x2, this.y2, this.score, this.classId);

  String get label => classId >= 0 && classId < kClassNames.length
      ? kClassNames[classId]
      : 'Class $classId';

  String get description =>
      kClassDescriptions[label] ?? 'No description available.';
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Error initializing cameras: $e');
  }
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
  final OnnxRuntime ort = OnnxRuntime();
  OrtSession? session;
  bool sessionReady = false;
  bool isLoading = false;
  File? imageFile;
  int imageWidth = 0;
  int imageHeight = 0;
  List<Detection> detections = [];
  double confThreshold = defaultConfThreshold;
  String? error;

  @override
  void initState() {
    super.initState();
    initializeSession();
  }

  @override
  void dispose() {
    session?.close();
    super.dispose();
  }

  Future<void> initializeSession() async {
    setState(() {
      sessionReady = false;
      error = null;
    });
    try {
      final options = OrtSessionOptions(
        providers: kIsWeb ? [OrtProvider.WEB_ASSEMBLY] : null,
      );
      session = await ort.createSessionFromAsset(modelAsset, options: options);
      if (mounted) {
        setState(() => sessionReady = true);
      }
    } catch (e) {
      final detail = e is PlatformException
          ? 'code: ${e.code}, message: ${e.message}'
          : e.toString();
      if (mounted) {
        setState(() => error = 'Failed to load model: $detail');
      }
    }
  }

  Future<void> pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1600,
      maxHeight: 1600,
    );
    if (picked == null) return;
    await processImage(File(picked.path));
  }

  Future<void> processImage(File file) async {
    setState(() {
      isLoading = true;
      error = null;
      detections = [];
      imageFile = file;
    });

    try {
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Unable to decode image');
      imageWidth = decoded.width;
      imageHeight = decoded.height;

      if (!sessionReady) await initializeSession();
      if (session == null) throw Exception('ONNX session is not ready');

      final inputTensor = await OrtValue.fromList(preprocess(bytes), [1, inputWidth, inputHeight, 3]);
      final outputs = await session!.run({inputName: inputTensor});
      final outputTensor = outputs[outputName];
      if (outputTensor == null) throw Exception('Model output not found');

      final flat = await outputTensor.asFlattenedList();
      final result = postprocess(flat);

      await inputTensor.dispose();
      await outputTensor.dispose();

      if (mounted) {
        setState(() {
          detections = result;
          isLoading = false;
        });
      }
    } catch (e) {
      final detail = e is PlatformException
          ? 'code: ${e.code}, message: ${e.message}'
          : e.toString();
      if (mounted) {
        setState(() {
          error = 'Detection failed: $detail';
          isLoading = false;
        });
      }
    }
  }

  Float32List preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image');
    final resized = img.copyResize(image, width: inputWidth, height: inputHeight);
    final data = Float32List(1 * inputWidth * inputHeight * 3);
    var idx = 0;
    for (var y = 0; y < inputHeight; y++) {
      for (var x = 0; x < inputWidth; x++) {
        final p = resized.getPixel(x, y);
        data[idx++] = p.r / 255.0;
        data[idx++] = p.g / 255.0;
        data[idx++] = p.b / 255.0;
      }
    }
    return data;
  }

  List<Detection> postprocess(List<dynamic> flat) {
    final result = <Detection>[];
    for (var i = 0; i < maxDetections; i++) {
      final base = i * 6;
      if (base + 5 >= flat.length) break;
      final x1 = (flat[base] as num).toDouble();
      final y1 = (flat[base + 1] as num).toDouble();
      final x2 = (flat[base + 2] as num).toDouble();
      final y2 = (flat[base + 3] as num).toDouble();
      final score = (flat[base + 4] as num).toDouble();
      final classId = ((flat[base + 5] as num).toDouble()).round();

      if (score <= 0 || score <= confThreshold) continue;

      var nx1 = x1;
      var ny1 = y1;
      var nx2 = x2;
      var ny2 = y2;
      if (x1 > 1 || y1 > 1 || x2 > 1 || y2 > 1) {
        nx1 /= inputWidth;
        ny1 /= inputHeight;
        nx2 /= inputWidth;
        ny2 /= inputHeight;
      }
      nx1 = nx1.clamp(0.0, 1.0);
      ny1 = ny1.clamp(0.0, 1.0);
      nx2 = nx2.clamp(0.0, 1.0);
      ny2 = ny2.clamp(0.0, 1.0);
      if (nx2 <= nx1 || ny2 <= ny1) continue;

      result.add(Detection(nx1, ny1, nx2, ny2, score, classId));
    }
    result.sort((a, b) => b.score.compareTo(a.score));
    return result;
  }

  Widget buildImageCanvas(BuildContext context) {
    if (imageFile == null || imageWidth == 0 || imageHeight == 0) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.image_outlined, size: 48, color: Colors.white54),
              SizedBox(height: 8),
              Text('No image selected', style: TextStyle(color: Colors.white54)),
            ],
          ),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final canvasW = constraints.maxWidth;
        final canvasH = constraints.maxHeight;
        final scaleW = canvasW / imageWidth;
        final scaleH = canvasH / imageHeight;
        final fitScale = scaleW < scaleH ? scaleW : scaleH;
        final dispW = imageWidth * fitScale;
        final dispH = imageHeight * fitScale;
        final offsetX = (canvasW - dispW) / 2;
        final offsetY = (canvasH - dispH) / 2;

        return Stack(
          fit: StackFit.expand,
          children: [
            Container(
              color: Colors.black,
              alignment: Alignment.center,
              child: Image.file(
                imageFile!,
                width: dispW,
                height: dispH,
                fit: BoxFit.fill,
              ),
            ),
            ...detections.map((det) {
              final color = classColors[det.classId % classColors.length];
              final left = (offsetX + det.x1 * dispW).clamp(0.0, canvasW - 1.0);
              final top = (offsetY + det.y1 * dispH).clamp(0.0, canvasH - 1.0);
              final width = ((det.x2 - det.x1) * dispW).clamp(0.0, canvasW - left);
              final height = ((det.y2 - det.y1) * dispH).clamp(0.0, canvasH - top);
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
                        top: -22,
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
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Tooth Condition Detection'),
        actions: [
          if (sessionReady)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: initializeSession,
              tooltip: 'Reload model',
            ),
        ],
      ),
      body: Column(
        children: [
          Card(
            margin: const EdgeInsets.all(12),
            elevation: 2,
            color: Colors.grey.shade900,
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
                          onPressed: isLoading ? null : () => pickImage(ImageSource.gallery),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          onPressed: isLoading ? null : () => pickImage(ImageSource.camera),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.videocam),
                          label: const Text('Live'),
                          onPressed: isLoading || !sessionReady
                              ? null
                              : () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => LiveDetectionPage(
                                        ort: ort,
                                        session: session!,
                                        confThreshold: confThreshold,
                                      ),
                                    ),
                                  );
                                },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Text('Confidence: ', style: TextStyle(color: Colors.white)),
                      Expanded(
                        child: Slider(
                          value: confThreshold,
                          min: 0.0,
                          max: 1.0,
                          divisions: 100,
                          label: (confThreshold * 100).round().toString(),
                          onChanged: isLoading
                              ? null
                              : (value) => setState(() => confThreshold = value),
                        ),
                      ),
                      Text('${(confThreshold * 100).round()}%', style: const TextStyle(color: Colors.white)),
                    ],
                  ),
                  if (error != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: Text(error!, style: const TextStyle(color: Colors.red)),
                    ),
                  if (imageFile != null && !isLoading)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: Row(
                        children: [
                          const Icon(Icons.sync, size: 18, color: Colors.white),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              'Detected ${detections.length} tooth condition(s)',
                              style: const TextStyle(fontWeight: FontWeight.w600, color: Colors.white),
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
              child: isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : buildImageCanvas(context),
            ),
          ),
          if (detections.isNotEmpty)
            SizedBox(
              height: 200,
              child: ListView.builder(
                itemCount: detections.length,
                itemBuilder: (context, i) {
                  final det = detections[i];
                  final color = classColors[det.classId % classColors.length];
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
                      style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                    ),
                    subtitle: Text(
                      '${det.description}\nConfidence: ${(det.score * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(color: Colors.white70),
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
      floatingActionButton: sessionReady && imageFile != null && !isLoading
          ? FloatingActionButton.extended(
              icon: const Icon(Icons.search),
              label: const Text('Detect'),
              onPressed: () => processImage(imageFile!),
            )
          : null,
    );
  }
}

class LiveDetectionPage extends StatefulWidget {
  final OnnxRuntime ort;
  final OrtSession session;
  final double confThreshold;

  const LiveDetectionPage({
    super.key,
    required this.ort,
    required this.session,
    required this.confThreshold,
  });

  @override
  State<LiveDetectionPage> createState() => _LiveDetectionPageState();
}

class _LiveDetectionPageState extends State<LiveDetectionPage> {
  CameraController? controller;
  bool isInitialized = false;
  bool isDetecting = false;
  List<Detection> detections = [];
  Timer? timer;

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

  Future<void> initializeCamera() async {
    if (cameras.isEmpty) return;
    controller = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await controller!.initialize();
      if (mounted) {
        setState(() {
          isInitialized = true;
        });
        startDetection();
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  void startDetection() {
    timer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (!isDetecting && isInitialized && controller != null && controller!.value.isInitialized) {
        captureAndDetect();
      }
    });
  }

  Future<void> captureAndDetect() async {
    if (isDetecting) return;
    isDetecting = true;
    try {
      final image = await controller!.takePicture();
      final bytes = await File(image.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded != null) {
        final inputTensor = await OrtValue.fromList(preprocess(bytes), [1, inputWidth, inputHeight, 3]);
        final outputs = await widget.session.run({inputName: inputTensor});
        final outputTensor = outputs[outputName];
        if (outputTensor != null) {
          final flat = await outputTensor.asFlattenedList();
          final result = postprocessLive(flat);
          if (mounted) {
            setState(() {
              detections = result;
            });
          }
          await inputTensor.dispose();
          await outputTensor.dispose();
        }
      }
      await File(image.path).delete();
    } catch (e) {
      debugPrint('Detection error: $e');
    } finally {
      isDetecting = false;
    }
  }

  Float32List preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image');
    final resized = img.copyResize(image, width: inputWidth, height: inputHeight);
    final data = Float32List(1 * inputWidth * inputHeight * 3);
    var idx = 0;
    for (var y = 0; y < inputHeight; y++) {
      for (var x = 0; x < inputWidth; x++) {
        final p = resized.getPixel(x, y);
        data[idx++] = p.r / 255.0;
        data[idx++] = p.g / 255.0;
        data[idx++] = p.b / 255.0;
      }
    }
    return data;
  }

  List<Detection> postprocessLive(List<dynamic> flat) {
    final result = <Detection>[];
    for (var i = 0; i < maxDetections; i++) {
      final base = i * 6;
      if (base + 5 >= flat.length) break;
      final x1 = (flat[base] as num).toDouble();
      final y1 = (flat[base + 1] as num).toDouble();
      final x2 = (flat[base + 2] as num).toDouble();
      final y2 = (flat[base + 3] as num).toDouble();
      final score = (flat[base + 4] as num).toDouble();
      final classId = ((flat[base + 5] as num).toDouble()).round();

      if (score <= 0 || score <= widget.confThreshold) continue;

      var nx1 = x1;
      var ny1 = y1;
      var nx2 = x2;
      var ny2 = y2;
      if (x1 > 1 || y1 > 1 || x2 > 1 || y2 > 1) {
        nx1 /= inputWidth;
        ny1 /= inputHeight;
        nx2 /= inputWidth;
        ny2 /= inputHeight;
      }
      nx1 = nx1.clamp(0.0, 1.0);
      ny1 = ny1.clamp(0.0, 1.0);
      nx2 = nx2.clamp(0.0, 1.0);
      ny2 = ny2.clamp(0.0, 1.0);
      if (nx2 <= nx1 || ny2 <= ny1) continue;

      result.add(Detection(nx1, ny1, nx2, ny2, score, classId));
    }
    result.sort((a, b) => b.score.compareTo(a.score));
    return result;
  }

  @override
  void dispose() {
    timer?.cancel();
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!isInitialized || controller == null) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: CircularProgressIndicator(color: Colors.white),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller!),
          LayoutBuilder(
            builder: (context, constraints) {
              final canvasW = constraints.maxWidth;
              final canvasH = constraints.maxHeight;
              return Stack(
                children: detections.map((det) {
                  final color = classColors[det.classId % classColors.length];
                  final left = (det.x1 * canvasW).clamp(0.0, canvasW - 1.0);
                  final top = (det.y1 * canvasH).clamp(0.0, canvasH - 1.0);
                  final width = ((det.x2 - det.x1) * canvasW).clamp(0.0, canvasW - left);
                  final height = ((det.y2 - det.y1) * canvasH).clamp(0.0, canvasH - top);
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
                            top: -22,
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
                }).toList(),
              );
            },
          ),
          Positioned(
            top: 40,
            left: 10,
            child: IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.white, size: 30),
              onPressed: () => Navigator.pop(context),
            ),
          ),
        ],
      ),
    );
  }
}
