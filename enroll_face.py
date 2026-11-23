"""Add a person to Aisana's face memory from photos (only with that person's consent).

    .venv/bin/python enroll_face.py "Waqar Ahmad" "President of Nazarbayev University" photo1.jpg [photo2.jpg ...]
    .venv/bin/python enroll_face.py --list
    .venv/bin/python enroll_face.py --forget "Waqar Ahmad"

Photos stay on the Jetson: keep them in faces_photos/ (not uploaded to GitHub).
Only the face vectors, the name and the title are stored, in the same database
as "Айсана, запомни меня". Restart Aisana afterwards (./stop.sh, ./start.sh).
A person with a title is greeted respectfully ("Hello, <name>, <title>. Welcome!")
and while such a guest is around she drops the sarcasm.
"""
import sys

import aisana_faces


def best_face(memory, path):
    import cv2
    image = cv2.imread(path)
    if image is None:
        raise SystemExit('Cannot read the photo: ' + path)
    height, width = image.shape[:2]
    if max(height, width) > 1600:                 # big photos: same framing, faster
        scale = 1600.0 / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    found = memory.faces(image)
    if not found:
        raise SystemExit('No face found in ' + path + ' (use a clear, frontal photo).')
    if len(found) > 1:
        print('  {}: {} faces, using the biggest one.'.format(path, len(found)))
    return found[0]


def main(argv):
    memory = aisana_faces.FaceMemory()
    if argv[:1] == ['--list']:
        for name in memory.names():
            title = memory.title(name)
            print('{}{}'.format(name, ' - ' + title if title else ''))
        return 0
    if argv[:1] == ['--forget'] and len(argv) == 2:
        print('Forgotten {} record(s).'.format(memory.forget(argv[1])))
        return 0
    if len(argv) < 3:
        print(__doc__)
        return 1
    name, title, photos = argv[0].strip(), argv[1].strip(), argv[2:]
    memory._load_nets()
    vectors = []
    for path in photos:
        face = best_face(memory, path)
        if face['name'] and face['name'] != name:
            print('  WARNING: {} looks like "{}" already in memory (distance {:.2f}).'.format(
                path, face['name'], face['distance']))
        vectors.append(face['vector'])
        print('  {}: face found.'.format(path))
    memory.remember(name, vectors)
    memory.set_title(name, title)
    print('Remembered {} ({}) from {} photo(s). Restart Aisana: ./stop.sh && ./start.sh'.format(
        name, title, len(vectors)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
