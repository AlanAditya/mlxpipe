/* mlxpipe face detection and face landmarks, as a plain C API.
 *
 * Self-contained: this header needs neither MLX nor OpenCV. Link against
 * libmlxpipe_face.dylib, which loads MediaPipe's face_landmarker.task bundle
 * and runs it on the GPU through MLX.
 *
 *   mlxpipe_face_detector_options options;
 *   mlxpipe_face_detector_default_options(&options);
 *   options.model_path = "tools/models/face_landmarker.task";
 *   mlxpipe_face_detector* detector = mlxpipe_face_detector_create(&options);
 *   if (!detector) fprintf(stderr, "%s\n", mlxpipe_last_error());
 *
 *   mlxpipe_image image = {pixels, width, height, bytes_per_row, MLXPIPE_PIXEL_BGRA};
 *   mlxpipe_face_detection faces[8];
 *   int n = mlxpipe_face_detector_detect(detector, &image, faces, 8);
 *   ...
 *   mlxpipe_face_detector_destroy(detector);
 *
 * Coordinates are normalized to [0, 1] relative to the image (x to the right,
 * y down). Functions that can fail return NULL or a negative value and set a
 * message for mlxpipe_last_error(). One instance must not be used from two
 * threads at once; separate instances are independent.
 */
#ifndef MLXPIPE_FACE_H_
#define MLXPIPE_FACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MLXPIPE_FACE_NUM_KEYPOINTS 6
#define MLXPIPE_FACE_NUM_LANDMARKS 478
#define MLXPIPE_FACE_NUM_BLENDSHAPES 52

typedef enum {
    MLXPIPE_PIXEL_RGB = 0,
    MLXPIPE_PIXEL_BGR = 1,
    MLXPIPE_PIXEL_RGBA = 2,
    MLXPIPE_PIXEL_BGRA = 3, /* CVPixelBuffer kCVPixelFormatType_32BGRA */
} mlxpipe_pixel_format;

/* 8-bit interleaved pixels. The data is only read during the call. */
typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int bytes_per_row; /* 0: tightly packed */
    mlxpipe_pixel_format format;
} mlxpipe_image;

typedef struct {
    float x;
    float y;
} mlxpipe_point2;

typedef struct {
    float x;
    float y;
    float z; /* depth, roughly the same scale as x; smaller is closer */
} mlxpipe_point3;

/* Returns the message of the last failed call on this thread ("" if none). */
const char* mlxpipe_last_error(void);

/* --- Face detection (BlazeFace short range): boxes and 6 keypoints ------- */

typedef struct {
    /* face_landmarker.task bundle or face_detector.tflite. */
    const char* model_path;
    float min_detection_confidence; /* default 0.5 */
    float min_suppression_threshold; /* default 0.3 */
    int max_faces; /* default -1: no limit */
} mlxpipe_face_detector_options;

typedef struct {
    float score;
    /* Bounding box: top-left corner and size. */
    float xmin;
    float ymin;
    float width;
    float height;
    /* Right eye, left eye, nose tip, mouth centre, right ear, left ear
     * (the subject's right and left). */
    mlxpipe_point2 keypoints[MLXPIPE_FACE_NUM_KEYPOINTS];
} mlxpipe_face_detection;

typedef struct mlxpipe_face_detector mlxpipe_face_detector;

void mlxpipe_face_detector_default_options(mlxpipe_face_detector_options* options);
mlxpipe_face_detector* mlxpipe_face_detector_create(const mlxpipe_face_detector_options* options);
void mlxpipe_face_detector_destroy(mlxpipe_face_detector* detector);

/* Detects faces in one image. Writes up to `capacity` detections to `out`,
 * highest score first, and returns how many faces were found (which may be
 * more than `capacity`), or -1 on error. */
int mlxpipe_face_detector_detect(mlxpipe_face_detector* detector, const mlxpipe_image* image,
                                 mlxpipe_face_detection* out, int capacity);

/* --- Face landmarks: 478-point mesh with irises, optional blendshapes ---- */

typedef struct {
    /* face_landmarker.task bundle. */
    const char* model_path;
    int max_faces; /* default 1 */
    float min_face_detection_confidence; /* default 0.5 */
    float min_face_presence_confidence; /* default 0.5 */
    float min_tracking_confidence; /* default 0.5 */
    int output_blendshapes; /* default 0 */
    int smooth_landmarks; /* default 1; video mode only */
} mlxpipe_face_landmarker_options;

typedef struct {
    /* Bounding box of all landmarks: top-left corner and size. Not clipped to
     * the image. */
    float xmin;
    float ymin;
    float width;
    float height;
    mlxpipe_point3 landmarks[MLXPIPE_FACE_NUM_LANDMARKS];
    /* Scores in [0, 1], named by mlxpipe_face_blendshape_name(); all zero
     * unless output_blendshapes is set. */
    float blendshapes[MLXPIPE_FACE_NUM_BLENDSHAPES];
} mlxpipe_face_landmarks;

typedef struct mlxpipe_face_landmarker mlxpipe_face_landmarker;

void mlxpipe_face_landmarker_default_options(mlxpipe_face_landmarker_options* options);
mlxpipe_face_landmarker* mlxpipe_face_landmarker_create(const mlxpipe_face_landmarker_options* options);
void mlxpipe_face_landmarker_destroy(mlxpipe_face_landmarker* landmarker);

/* Image mode: every call runs face detection. Returns the number of faces
 * (writing up to `capacity`), or -1 on error. */
int mlxpipe_face_landmarker_detect(mlxpipe_face_landmarker* landmarker, const mlxpipe_image* image,
                                   mlxpipe_face_landmarks* out, int capacity);

/* Video mode: tracks faces across frames and only runs face detection when
 * fewer than max_faces are tracked. Timestamps must increase. */
int mlxpipe_face_landmarker_detect_for_video(mlxpipe_face_landmarker* landmarker, const mlxpipe_image* image,
                                             int64_t timestamp_ms, mlxpipe_face_landmarks* out, int capacity);

/* "_neutral", "browDownLeft", ... for index 0..51; NULL otherwise. */
const char* mlxpipe_face_blendshape_name(int index);

/* Groups of mesh landmarks, from MediaPipe's face mesh connection tables.
 * Left and right are the subject's: in an unmirrored image the left eye is on
 * the image's right. */
typedef enum {
    MLXPIPE_FACE_REGION_LEFT_EYE = 0,
    MLXPIPE_FACE_REGION_RIGHT_EYE,
    MLXPIPE_FACE_REGION_LEFT_EYEBROW,
    MLXPIPE_FACE_REGION_RIGHT_EYEBROW,
    MLXPIPE_FACE_REGION_LEFT_IRIS,
    MLXPIPE_FACE_REGION_RIGHT_IRIS,
    MLXPIPE_FACE_REGION_NOSE,
    MLXPIPE_FACE_REGION_LIPS,
    MLXPIPE_FACE_REGION_FACE_OVAL,
    MLXPIPE_FACE_NUM_REGIONS
} mlxpipe_face_region;

/* Sets *indices to the region's landmark indices (into
 * mlxpipe_face_landmarks.landmarks, in contour order, each listed once) and
 * returns how many there are. The array stays valid while the library is
 * loaded. Returns 0 and sets NULL for an unknown region. */
int mlxpipe_face_region_indices(mlxpipe_face_region region, const int** indices);

#ifdef __cplusplus
}
#endif

#endif /* MLXPIPE_FACE_H_ */
