# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from tcdm.envs import asset_abspath
from .base import MjModel
from dm_control import mjcf
import numpy as np


class ObjectModel(MjModel):
    def __init__(self, base_pos, base_quat, mjcf_model):
        self._base_pos = np.array(base_pos).copy()
        self._base_quat = np.array(base_quat).copy()
        if isinstance(mjcf_model, str):
            mjcf_model = mjcf.from_path(mjcf_model)
            mjcf_model.worldbody.all_children()[0].pos = self._base_pos
            mjcf_model.worldbody.all_children()[0].quat = self._base_quat
        super().__init__(mjcf_model)

    @property
    def start_pos(self):
        return self._base_pos.copy().astype(np.float32)

    @property
    def start_ori(self):
        return self._base_quat.copy().astype(np.float32)


class DAPGHammerObject(ObjectModel):
    def __init__(self, pos=[0, -0.2, 0.035], 
                    quat=[0.707388, 0.706825, 0, 0]):
        xml_path = asset_abspath('objects/dapg_hammer.xml')
        super().__init__(pos, quat, xml_path)


def object_generator(path):
    class __XMLObj__(ObjectModel):
        def __init__(self, pos=[0.0, 0.0, 0.2], 
                       quat=[1, 0, 0, 0]):
            xml_path = asset_abspath(path)
            super().__init__(pos, quat, xml_path)
    return __XMLObj__


AirplaneObject = object_generator('objects/airplane.xml')
AlarmClockObject = object_generator('objects/alarmclock.xml')
AppleObject = object_generator('objects/apple.xml')
BananaObject = object_generator('objects/banana.xml')
BinocularsObject = object_generator('objects/binoculars.xml')
BodyObject = object_generator('objects/body.xml')
BowlObject = object_generator('objects/bowl.xml')
CameraObject = object_generator('objects/camera.xml')
CoffeeCanObject = object_generator('objects/coffeecan.xml')
CoffeeMugObject = object_generator('objects/coffeemug.xml')
CrackerBoxObject = object_generator('objects/crackerbox.xml')
CubeLargeObject = object_generator('objects/cubelarge.xml')
CubeMediumObject = object_generator('objects/cubemedium.xml')
CubeMiddleObject = object_generator('objects/cubemiddle.xml')
CubeSmallObject = object_generator('objects/cubesmall.xml')
CupObject = object_generator('objects/cup.xml')
CylinderLargeObject = object_generator('objects/cylinderlarge.xml')
CylinderMediumObject = object_generator('objects/cylindermedium.xml')
CylinderSmallObject = object_generator('objects/cylindersmall.xml')
DoorObject = object_generator('objects/door.xml')
DoorknobObject = object_generator('objects/doorknob.xml')
DuckObject = object_generator('objects/duck.xml')
ElephantObject = object_generator('objects/elephant.xml')
EyeglassesObject = object_generator('objects/eyeglasses.xml')
FlashlightObject = object_generator('objects/flashlight.xml')
FluteObject = object_generator('objects/flute.xml')
FryingPanObject = object_generator('objects/fryingpan.xml')
GameControllerObject = object_generator('objects/gamecontroller.xml')
HammerObject = object_generator('objects/hammer.xml')
HandObject = object_generator('objects/hand.xml')
HeadphonesObject = object_generator('objects/headphones.xml')
KnifeObject = object_generator('objects/knife.xml')
LightBulbObject = object_generator('objects/lightbulb.xml')
MouseObject = object_generator('objects/mouse.xml')
MugObject = object_generator('objects/mug.xml')
NailObject = object_generator('objects/nail.xml')
PhoneObject = object_generator('objects/phone.xml')
PiggyBankObject = object_generator('objects/piggybank.xml')
PyramidLargeObject = object_generator('objects/pyramidlarge.xml')
PyramidMediumObject = object_generator('objects/pyramidmedium.xml')
PyramidSmallObject = object_generator('objects/pyramidsmall.xml')
RubberDuckObject = object_generator('objects/rubberduck.xml')
ScissorsObject = object_generator('objects/scissors.xml')
SphereLargeObject = object_generator('objects/spherelarge.xml')
SphereMediumObject = object_generator('objects/spheremedium.xml')
SphereSmallObject = object_generator('objects/spheresmall.xml')
StampObject = object_generator('objects/stamp.xml')
StanfordBunnyObject = object_generator('objects/stanfordbunny.xml')
StaplerObject = object_generator('objects/stapler.xml')
TableObject = object_generator('objects/table.xml')
TeapotObject = object_generator('objects/teapot.xml')
ToothbrushObject = object_generator('objects/toothbrush.xml')
ToothpasteObject = object_generator('objects/toothpaste.xml')
TorusLargeObject = object_generator('objects/toruslarge.xml')
TorusMediumObject = object_generator('objects/torusmedium.xml')
TorusSmallObject = object_generator('objects/torussmall.xml')
TrainObject = object_generator('objects/train.xml')
WatchObject = object_generator('objects/watch.xml')
WaterBottleObject = object_generator('objects/waterbottle.xml')
WineGlassObject = object_generator('objects/wineglass.xml')
WristwatchObject = object_generator('objects/wristwatch.xml')


_OBJ_LOOKUP = {
                'airplane': AirplaneObject,
                'alarmclock': AlarmClockObject,
                'apple': AppleObject,
                'banana': BananaObject,
                'binoculars': BinocularsObject,
                'body': BodyObject,
                'bowl': BowlObject,
                'camera': CameraObject,
                'coffeecan': CoffeeCanObject,
                'coffeemug': CoffeeMugObject,
                'crackerbox': CrackerBoxObject,
                'cubelarge': CubeLargeObject,
                'cubemedium': CubeMediumObject,
                'cubemiddle': CubeMiddleObject,
                'cubesmall': CubeSmallObject,
                'cup': CupObject,
                'cylinderlarge': CylinderLargeObject,
                'cylindermedium': CylinderMediumObject,
                'cylindersmall': CylinderSmallObject,
                'dapghammer': DAPGHammerObject,
                'door': DoorObject,
                'doorknob': DoorknobObject,
                'duck': DuckObject,
                'elephant': ElephantObject,
                'eyeglasses': EyeglassesObject,
                'flashlight': FlashlightObject,
                'flute': FluteObject,
                'fryingpan': FryingPanObject,
                'gamecontroller': GameControllerObject,
                'hammer': HammerObject,
                'hand': HandObject,
                'headphones': HeadphonesObject,
                'knife': KnifeObject,
                'lightbulb': LightBulbObject,
                'mouse': MouseObject,
                'mug': MugObject,
                'nail': NailObject,
                'phone': PhoneObject,
                'piggybank': PiggyBankObject,
                'pyramidlarge': PyramidLargeObject,
                'pyramidmedium': PyramidMediumObject,
                'pyramidsmall': PyramidSmallObject,
                'rubberduck': RubberDuckObject,
                'scissors': ScissorsObject,
                'spherelarge': SphereLargeObject,
                'spheremedium': SphereMediumObject,
                'spheresmall': SphereSmallObject,
                'stamp': StampObject,
                'stanfordbunny': StanfordBunnyObject,
                'stapler': StaplerObject,
                'table': TableObject,
                'teapot': TeapotObject,
                'toothbrush': ToothbrushObject,
                'toothpaste': ToothpasteObject,
                'toruslarge': TorusLargeObject,
                'torusmedium': TorusMediumObject,
                'torussmall': TorusSmallObject,
                'train': TrainObject,
                'watch': WatchObject,
                'waterbottle': WaterBottleObject,
                'wineglass': WineGlassObject,
                'wristwatch': WristwatchObject,
             }


def get_object(name):
    if name not in _OBJ_LOOKUP:
        raise NotImplementedError("Obj {} Not Implemented!".format(name))
    return _OBJ_LOOKUP[name]
