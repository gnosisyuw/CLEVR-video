# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
                    help="JSON file defining objects, materials, sizes, and colors. " +
                         "The \"colors\" field maps from CLEVR color names to RGB values; " +
                         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                         "rescale object models; the \"materials\" and \"shapes\" fields map " +
                         "from CLEVR material and shape names to .blend files in the " +
                         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=4, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=6, type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.55, type=float,
                    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
                    help="Along all cardinal directions (left, right, front, back), all " +
                         "objects will be at least this distance apart. This makes resolving " +
                         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
                    help="All objects will have at least this many visible pixels in the " +
                         "final rendered images; this ensures that no objects are fully " +
                         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
                    help="The number of times to try placing an object before giving up and " +
                         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../outputv3/images/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../outputv3/scenes/',
                    help="The directory where output JSON scene structures will be stored. " +
                         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../outputv3/CLEVR_scenes.json',
                    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='outputv3/blendfiles',
                    help="The directory where blender scene files will be stored, if the " +
                         "user requested that these files be saved using the " +
                         "--save_blendfiles flag; in this case it will be created if it does " +
                         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
                    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                         "each generated image to be stored in the directory specified by " +
                         "the --output_blend_dir flag. These files are not saved by default " +
                         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
                    default="Creative Commons Attribution (CC-BY 4.0)",
                    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file; " +
                         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=256, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=256, type=int,
                    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")


def main(args):
    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    # img_template = '%s%%0%dd_%%%%02d.png' % (prefix, num_digits)
    # print(img_template,prefix,num_digits)
    img_template = '%s%%0%dd' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    img_template = os.path.join(args.output_image_dir, img_template)
    scene_template = os.path.join(args.output_scene_dir, scene_template)
    blend_template = os.path.join(args.output_blend_dir, blend_template)

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir)

    all_scene_paths = []
    for i in range(args.num_images):
        img_path = img_template % (i + args.start_idx)
        # os.makedirs(img_path,exist_ok=True)
        scene_path = scene_template % (i + args.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = blend_template % (i + args.start_idx)
        num_objects = random.randint(args.min_objects, args.max_objects)
        render_scene(args,
                     num_objects=num_objects,
                     output_index=(i + args.start_idx),
                     output_split=args.split,
                     output_image=img_path,
                     output_scene=scene_path,
                     output_blendfile=blend_path,
                     )

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)


def render_scene(args,
                 num_objects=5,
                 output_index=0,
                 output_split='none',
                 output_image='render.png',
                 output_scene='render_json',
                 output_blendfile=None,
                 ):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    index = 0
    save_template = '/%02d.png'
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image + save_template % (index)
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender c1d the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    objects, blender_objects, reference = add_random_objects(scene_struct, num_objects, args, camera)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    # while True:
    #     try:
    #         bpy.ops.render.render(write_still=True)
    #         break
    #     except Exception as e:
    #         print(e)




    actmap = {'SmoothCylinder': [0, 5], 'SmoothCube': [0, 1, 2, 3, 4, 5]}
    move_list = {0: Vector((0.0, 0.0, 2.4)), 1: Vector((0.8, 0.0, 2.4)), 2: Vector((0.0, 0.8, 2.4)),
                 3: Vector((0.0, -0.8, 2.4)), 4: Vector((-0.8, 0.0, 2.4))}
    drop_list = {0: Vector((0.0, 0.0, -1.6)), 1: Vector((0.0, 0.0, -2.4)), 2: Vector((0.0, 0.0, -2.4)),
                 3: Vector((0.0, 0.0, -2.4)), 4: Vector((0.0, 0.0, -2.4))}
    lift_vect =  Vector((0.0, 0.0, 2.4))
    item1 = bpy.data.objects[reference[0]]
    item2 = bpy.data.objects[reference[1]]
    item3 = bpy.data.objects[reference[2]]

    x1, y1, z1 = item1.location
    x2, y2, z2 = item2.location
    x3, y3, z3 = item3.location

    s1 = item1.name.split('_')[0]
    s2 = item2.name.split('_')[0]
    s3 = item3.name.split('_')[0]
    a1 = item1.rotation_euler[2]
    a2 = item2.rotation_euler[2]
    a3 = item3.rotation_euler[2]

    lift_t = 2
    current_t = 1
    move_t = 2
    drop_t = 2

    scene = bpy.data.scenes["Scene"]
    # 1+7+3+7+6+3+7+6+3

    l1 = actmap[s1].copy()
    l2 = actmap[s2].copy()
    l3 = actmap[s3].copy()

    c1, c2, c3 = 0, 0, 0

    scene.frame_start = 1

    # 1st align
    if len(l1) == 2:
        c1  = random.randint(0, 2)
        l1 = [l1[0] + c1, l1[1] - c1]

    item1.location = (x1, y1, z1)
    item1.keyframe_insert(data_path='location', frame = current_t)
    item1.location = item1.location + lift_vect
    item1.keyframe_insert(data_path='location', frame = current_t + lift_t)

    item1.rotation_euler = item1.rotation_euler
    item1.keyframe_insert(data_path='rotation_euler', frame = current_t)

    if len(l1) == 2:
        if c1 == 0:
            item1.rotation_euler = item1.rotation_euler
        elif c1 == 1:
            item1.rotation_euler = (0, 1.5708, 0)
        elif c1 == 2:
            item1.rotation_euler = (1.5708, 0, 0)
    else:
        item1.rotation_euler = (0, 0, a1//1.57084*1.5708)
    item1.keyframe_insert(data_path='rotation_euler', frame=current_t + lift_t)
    current_t += lift_t
    # 1st drop
    item1.location = (x1, y1, z1)
    item1.keyframe_insert(data_path='location', frame=current_t + drop_t)
    current_t += drop_t
    # 1st update
    if 0 in l1:
        l1.remove(0)

    # second align
    if len(l2) == 2:
        if len(l1)<=2:
            c2 = c1
        else:
            c2  = random.randint(0, 2)
        l2 = [l2[0] + c2, l2[1] - c2]
    item2.location = (x2, y2, z2)
    item2.keyframe_insert(data_path='location', frame=current_t)
    item2.location = item2.location + lift_vect
    item2.keyframe_insert(data_path='location', frame=current_t + lift_t)

    item2.rotation_euler = item2.rotation_euler
    item2.keyframe_insert(data_path='rotation_euler', frame=current_t)

    if len(l2) == 2:
        if c2== 0:
            item2.rotation_euler = item2.rotation_euler
        elif c2 == 1:
            item2.rotation_euler = (0, 1.5708, 0)
        elif c2 == 2:
            item2.rotation_euler = (1.5708, 0, 0)
    else:
        item2.rotation_euler = (0, 0, a2//1.57084*1.5708)

    item2.keyframe_insert(data_path='rotation_euler', frame=current_t+lift_t)
    current_t+=lift_t

    # 2nd move
    pos = 0
    if len(l1) == 1:
        pos = 0
    elif len(l1) == 2:
        if c1 ==1:
            if random.randint(0,1) == 0:
                pos = 1
            else:
                pos = 4
        if c1 == 2:
            if random.randint(0, 1) == 0:
                pos = 2
            else:
                pos = 3
    else:
        new_list = l2.copy()
        if 5 in new_list:
            new_list.remove(5)
        pos = random.choice(new_list)

    item2.location =item1.location + move_list[pos]
    item2.keyframe_insert(data_path='location', frame=current_t + move_t)
    current_t += move_t

    # 2nd drop
    item2.location = item2.location + drop_list[pos]
    item2.keyframe_insert(data_path='location', frame=current_t+drop_t)
    current_t += drop_t

    # 2nd update
    l1.remove(5 - pos)
    l2.remove(pos)
    if pos == 0:
        l2 = [5]
    if 0 in l2:
        l2.remove(0)
    att_loc = ['1_' + str(x) for x in l1] +  ['2_' + str(x) for x in l2]
    action_1 = pos


    # 3rd align
    if len(l3) == 2:
        if len(att_loc) <= 2:
            c3 = c1
        else:
            c3 = random.randint(0, 2)
        l3 = [l3[0] + c3, l3[1] - c3]
    item3.location = (x3, y3, z3)
    item3.keyframe_insert(data_path='location', frame=current_t)
    item3.location = item3.location + lift_vect
    item3.keyframe_insert(data_path='location', frame=current_t + lift_t)

    item3.rotation_euler = item3.rotation_euler
    item3.keyframe_insert(data_path='rotation_euler', frame=current_t)

    if len(l3) == 2:
        if c3 == 0:
            item3.rotation_euler = item3.rotation_euler
        elif c3 == 1:
            item3.rotation_euler = (0, 1.5708, 0)
        elif c3 == 2:
            item3.rotation_euler = (1.5708, 0, 0)
    else:
        item3.rotation_euler = (0, 0, a3 // 1.57084 * 1.5708)

    item3.keyframe_insert(data_path='rotation_euler', frame=current_t + lift_t)
    current_t += lift_t

    # 3rd move
    pos = 0
    possible_ob = []
    if len(att_loc) == 1:
        pos = 0
    elif len(att_loc) == 2:
        if c1 == 1:
            if random.randint(0, 1) == 0:
                pos = 1
            else:
                pos = 4
        if c1 == 2:
            if random.randint(0, 1) == 0:
                pos = 2
            else:
                pos = 3
    else:
        new_list = l3.copy()
        if 5 in new_list:
            new_list.remove(5)
        pos = random.choice(new_list)

    for p in att_loc:
        if 5 - pos == int(p.split('_')[1]):
            possible_ob.append(int(p.split('_')[0]))
    use_location,record = (item1.location,1) if random.choice(possible_ob)== 1 else (item2.location,2)

    # use_location = item1.location if random.choice(possible_ob)== 1 else item2
    add_vec = Vector((0,0,-0.8)) if len(att_loc) == 1 else Vector((0,0,0))

    item3.location = use_location + move_list[pos] + add_vec

    item3.keyframe_insert(data_path='location', frame=current_t + move_t)
    current_t += move_t

    # 3rd drop

    item3.location = item3.location + drop_list[pos] - add_vec
    item3.keyframe_insert(data_path='location', frame=current_t + drop_t)
    current_t += drop_t



    # with open('/home/stevelab2/Documents/stacking/outputv3/temp.json', 'w+') as f:
    #     json.dump(att_loc,f)
    #     json.dump([record,pos, list(item3.location)], f)
    #     json.dump(output_image, f)
    # #     json.dump(new_list,f)

    scene_struct['actions'] = [action_1 ,record, pos]


    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


    scene.frame_end = current_t

    scene.render.filepath = output_image + ".avi"
    scene.render.image_settings.file_format = "AVI_JPEG"
    bpy.ops.render.render(animation=True)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    reference = []
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    pair_list = [(x,y) for x in object_mapping for y in color_name_to_rgba.keys()]


    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, 'r') as f:
            shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        # Choose a random size
        size_name, r = random.choice(size_mapping)

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(margin, args.margin, direction_name)
                        print('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose random color and shape
        if shape_color_combos is None:
            # obj_name, obj_name_out = random.choice(object_mapping)
            # color_name, rgba = random.choice(list(color_name_to_rgba.items()))
            the_chosen = random.choice(pair_list)
            obj_ppp, color_name= the_chosen
            obj_name, obj_name_out = obj_ppp
            rgba = color_name_to_rgba[color_name]
            pair_list.remove(the_chosen)
        else:
            obj_name_out, color_choices = random.choice(shape_color_combos)
            color_name = random.choice(color_choices)
            obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
            rgba = color_name_to_rgba[color_name]

        # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = math.radians(360) * random.random()
        if obj_name == 'SmoothCylinder':
            theta = 0.0

        # Actually add the object to the scene
        ref = utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        reference.append(ref)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach a random material
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })

    # Check that all objects are at least partially visible in the rendered image
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

    # with open('/home/wei/PycharmProjects/stacking/output/images/save.json', 'a+') as f:
    #     json.dump(reference,f)
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)

    return objects, blender_objects, reference


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3])
                          for i in range(0, len(p), 4))
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

