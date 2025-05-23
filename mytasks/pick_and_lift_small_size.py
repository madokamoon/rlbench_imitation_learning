from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class PickAndLiftSmallSize(Task):


    # ----------------------增加的内容------------------------
    # 用于改回原名
    def __init__(self, pyrep, robot, name: str = 'pick_and_lift'):
        super().__init__(pyrep, robot, name)
    # ----------------------增加的内容------------------------

    def init_task(self) -> None:



        self.target_block = Shape('pick_and_lift_target')
        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(2)]
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])

        self.success_detector = ProximitySensor('pick_and_lift_success')

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])




    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self.boundary.clear()
        # ----------------------增加的内容------------------------
        # 重新初始化 self.boundary._boundaries  
        # 重新计算 self.boundary._probabilities
        from rlbench.backend.spawn_boundary import BoundingBox
        areas= []
        for bo in self.boundary._boundaries:
            # print('原始的_boundary_bbox:', bo._boundary.get_bounding_box())
            #  [-0.21500001847743988, 0.21500001847743988, -0.2824999988079071, 0.2824999988079071, -0.0, 0.0]
            minx, maxx, miny, maxy, minz, maxz = [-0.15, 0.15, -0.15, 0.15, -0.0, 0.0]
            bo._boundary_bbox = BoundingBox(minx, maxx, miny, maxy, minz, maxz)
            # print('修改为_boundary_bbox:', bo._boundary_bbox)
            height = np.abs(maxz - minz)
            if height == 0:
                height = 1.0
                bo._is_plane = True
            bo._area = np.abs(maxx - minx) * np.abs(maxy - miny) * height
            # print('修改_area:', bo._boundary_bbox)
            areas.append(bo.get_area())
        self.boundary._probabilities = np.array(areas) / np.sum(areas)

        # ----------------------增加的内容------------------------
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for block in [self.target_block] + self.distractors:
            self.boundary.sample(block, min_distance=0.1)

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([self.target_block.get_position(), self.success_detector.get_position()], 0)
