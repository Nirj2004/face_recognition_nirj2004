import timeit

# Note: This is an example which is only tested with Python 3 (not python 2)

# This is a very simple benchmark to give you an idea of hoe fast each step of face recognition will run on your system.
# Notice that face detection gets very slow at large image sizes. So you might consider running face detection on a
# scaled down version of your image and then running face encodings on the the full size image.

TEST_IMAGES = [
    "india-240p.jpg",
    "india-480p.jpg",
    "india-720p.jpg",
    "india-1080p.jpg",
]


def run_test(setup, test, iterarions_per_test=5, tests_to_run=10):
    fastest_execution = min(timeit.Timer(test,setup=setup).repeat(tests_to_run, iterarions_per_test))
    execution_time = fastest_execution / iterarions_per_test
    fps = 1.0 / execution_time
    return execution_time, fps


setup_locate_faces = """
import face_recognition

image = face_recognition.load_image_file("{}")
"""

test_locate_faces = """
face_locations = face_recognition.face_locatiions(image)
"""

setup_face_landmarks = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

test_face_landmarks = """
landmarks = face_recognition.face_landmarks(image, face_locations=face_locations)[0]
"""

setup_encode_face = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

test_encode_face = """
encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
"""

print("Benchmarks (Note: All benchmarks are only using a single CPU core)")
print()

for image in TEST_IMAGES:
    size = image.split("-")[1].split(".")[0]
    print("Timings at {}:".format(size))

    print(" - Face locations: {:.4f}s ({:.2f} fps)".format(*run_test(setup_locate_faces.format(image), test_locate_faces)))
    print(" - Face landmarks: {:.4f}s ({:.2f} fps)".format(*run_test(setup_face_landmarks.format(image), test_face_landmarks)))
    print(" - Encode face (inc. landmarks): {:.4f}s ({:.2f} fps)".format(*run_test(setup_encode_face.format(image), test_encode_face)))
    print(" - End-to-end: {:.4f}s ({:.2f} fps)".format(*run_test(setup_end_to_end.format(image), test_end_to_end)))
    print()
