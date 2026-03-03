from supertem.config import SuperTEMContext
from supertem.utils import setup_session
from supertem.structures.base import *

# 1. Create the Context
ctx = SuperTEMContext.testing(tmp_path=Path("C:\\Users\Acer\PycharmProjects\SuperTem\offline_testing"))

# 2. Initialize the Session
scope, settings = setup_session(context=ctx, manufacturer="JEOL", profile_name="pyjem_offline", offline=True)

print(scope.get_beam_settings().mode)
target = BeamSettings(mode='STEM')
request = BeamControlRequest(target=target)
scope.execute_beam_control(request=request)
print(scope.get_beam_settings().mode)


def change_mode():
    print(scope.get_beam_settings().mode)
    target = BeamSettings(mode='STEM')
    request = BeamControlRequest(target=target)
    scope.execute_beam_control(request=request)
    print(scope.get_beam_settings().mode)

def stage_related():
    stage_target = StagePosition(x=10, y=10)
    print(scope.get_stage_position().to_dict())
    scope.execute_stage_move(request=StageMoveRequest(target=stage_target))
    print(scope.get_stage_position().to_dict())
    scope.execute_stage_move(request=StageMoveRequest(target=stage_target, relative=True))
    print(scope.get_stage_position().to_dict())
    scope.execute_stage_control(request=StageControlRequest(action='HOME'))
    print(scope.get_stage_position().to_dict())
    scope.execute_stage_move(request=StageMoveRequest(target=StagePosition(x=0, y=0)))
    print(scope.get_stage_position().to_dict())
    stage_target_big = StagePosition(x=80000, y=80000)
    scope.execute_stage_move(request=StageMoveRequest(target=stage_target_big))
    print(scope.get_stage_position().to_dict())

def beam_related():
    beam_target = BeamSettings(voltage=80, spot_size=3)
    print(scope.get_beam_settings().to_dict())
    scope.execute_beam_control(request=BeamControlRequest(target=beam_target))
    print(scope.get_beam_settings().to_dict())
    scope.execute_beam_control(request=BeamControlRequest(target=BeamSettings(extra=Extras(vendor={'JEOL': {'alpha_index': 3}}))))
    print(scope.get_beam_settings().to_dict())

def projection_related():
    projection_target = ProjectionSettings(magnification=10000, screen_position='UP')
    print(scope.get_projection_settings().to_dict())
    scope.execute_projection_control(request=ProjectionControlRequest(target=projection_target))
    print(scope.get_projection_settings().to_dict())
    # need to check the actual eos table and also the index

def detector_related():
    detector_target = DetectorSettings(detector_id='camera', exposure = Q_(50, Units.MS), gain_index=4095, extra=Extras(vendor={'JEOL': {'ScanRotation': 3}}))
    print(scope.get_detector_settings(detector_id='camera').to_dict())
    scope.execute_detector_control(request=DetectorControlRequest(target=detector_target))
    print(scope.get_detector_settings(detector_id='camera').to_dict())

def aperture_related():
    aperture_target = ApertureSettings(aperture_id='CL1', size_index=1, position=Point(x=10, y=10))
    print(scope.get_aperture_settings(aperture_id='CL1').to_dict())
    scope.execute_aperture_control(request=ApertureControlRequest(target=aperture_target, relative=True))
    print(scope.get_aperture_settings(aperture_id='CL1').to_dict())


if __name__ == '__main__':
    print("Success")
    # save_live_config(microscope=scope, context=ctx, name="pyjem_offline")
    # change_mode()
    # stage_related()
    # beam_related()
    # projection_related()
    # aperture_related()
    # detector_related()