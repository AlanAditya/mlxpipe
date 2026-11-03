/* The face C API (include/mlxpipe_face.h), compiled as C and linked against
 * libmlxpipe_face.dylib only: options, error reporting, empty results. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlxpipe_face.h"

static int failures = 0;
#define EXPECT(cond)                                                             \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main(void) {
    const char* task = MLXPIPE_SOURCE_DIR "/tools/models/face_landmarker.task";

    /* Noise frame, BGRA with padded rows. */
    const int width = 640, height = 480, stride = width * 4 + 64;
    uint8_t* pixels = malloc((size_t)stride * height);
    srand(1);
    for (int i = 0; i < stride * height; ++i) pixels[i] = (uint8_t)rand();
    mlxpipe_image image = {pixels, width, height, stride, MLXPIPE_PIXEL_BGRA};

    mlxpipe_face_detector_options detector_options;
    mlxpipe_face_detector_default_options(&detector_options);
    EXPECT(detector_options.max_faces == -1);
    EXPECT(detector_options.min_detection_confidence == 0.5f);

    /* Errors come back as NULL / -1 with a message. */
    detector_options.model_path = "/nonexistent.task";
    EXPECT(mlxpipe_face_detector_create(&detector_options) == NULL);
    EXPECT(strlen(mlxpipe_last_error()) > 0);
    EXPECT(mlxpipe_face_detector_create(NULL) == NULL);

    detector_options.model_path = task;
    mlxpipe_face_detector* detector = mlxpipe_face_detector_create(&detector_options);
    EXPECT(detector != NULL);
    EXPECT(strcmp(mlxpipe_last_error(), "") == 0);
    mlxpipe_face_detection detections[4];
    EXPECT(mlxpipe_face_detector_detect(detector, &image, detections, 4) == 0);
    EXPECT(mlxpipe_face_detector_detect(detector, NULL, detections, 4) == -1);
    mlxpipe_image too_narrow = {pixels, width, height, width, MLXPIPE_PIXEL_BGRA};
    EXPECT(mlxpipe_face_detector_detect(detector, &too_narrow, detections, 4) == -1);
    mlxpipe_face_detector_destroy(detector);

    mlxpipe_face_landmarker_options landmarker_options;
    mlxpipe_face_landmarker_default_options(&landmarker_options);
    EXPECT(landmarker_options.max_faces == 1);
    EXPECT(landmarker_options.smooth_landmarks == 1);
    landmarker_options.model_path = task;
    landmarker_options.output_blendshapes = 1;
    mlxpipe_face_landmarker* landmarker = mlxpipe_face_landmarker_create(&landmarker_options);
    EXPECT(landmarker != NULL);
    mlxpipe_face_landmarks faces[1];
    EXPECT(mlxpipe_face_landmarker_detect(landmarker, &image, faces, 1) == 0);
    EXPECT(mlxpipe_face_landmarker_detect_for_video(landmarker, &image, 100, faces, 1) == 0);
    EXPECT(mlxpipe_face_landmarker_detect_for_video(landmarker, &image, 133, faces, 1) == 0);
    /* Timestamps must increase. */
    EXPECT(mlxpipe_face_landmarker_detect_for_video(landmarker, &image, 133, faces, 1) == -1);
    mlxpipe_face_landmarker_destroy(landmarker);

    EXPECT(strcmp(mlxpipe_face_blendshape_name(0), "_neutral") == 0);
    EXPECT(strcmp(mlxpipe_face_blendshape_name(51), "noseSneerRight") == 0);
    EXPECT(mlxpipe_face_blendshape_name(52) == NULL);

    /* Region tables: unique indices into the 478-point mesh. */
    const int* idx = NULL;
    EXPECT(mlxpipe_face_region_indices(MLXPIPE_FACE_REGION_LEFT_EYE, &idx) == 16 && idx && idx[0] == 263);
    EXPECT(mlxpipe_face_region_indices(MLXPIPE_FACE_REGION_RIGHT_EYEBROW, &idx) == 10 && idx[0] == 46);
    EXPECT(mlxpipe_face_region_indices(MLXPIPE_FACE_REGION_NOSE, &idx) == 24 && idx[0] == 168);
    EXPECT(mlxpipe_face_region_indices(MLXPIPE_FACE_REGION_FACE_OVAL, &idx) == 36);
    for (int r = 0; r < MLXPIPE_FACE_NUM_REGIONS; ++r) {
        int n = mlxpipe_face_region_indices((mlxpipe_face_region)r, &idx);
        EXPECT(n > 0);
        for (int k = 0; k < n; ++k) EXPECT(idx[k] >= 0 && idx[k] < MLXPIPE_FACE_NUM_LANDMARKS);
    }
    EXPECT(mlxpipe_face_region_indices(MLXPIPE_FACE_NUM_REGIONS, &idx) == 0 && idx == NULL);

    free(pixels);
    if (failures) fprintf(stderr, "%d failure(s)\n", failures);
    else printf("PASS\n");
    return failures ? 1 : 0;
}
