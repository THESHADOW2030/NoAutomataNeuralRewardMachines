import torch
import argparse
import cv2

import utils


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="GridWorld-fixed-v1", type=str)
parser.add_argument("--render_mode", default="human", choices=["human", "terminal"])
parser.add_argument("--input_type", default="keyboard", choices=["keyboard", "terminal"])
parser.add_argument("--formula_id", default=0, type=int)
parser.add_argument("--sampler", default="Dataset_e54test_no-shuffle", type=str)
args = parser.parse_args()


# build environment
env = utils.make_env(
    args.env,
    progression_mode = "full",
    ltl_sampler = args.sampler,
    seed = 1,
    obs_size = (56,56)
)
env.env.render_mode = args.render_mode

if "GridWorld" in args.env:
    str_to_action = {"s":0,"d":1,"w":2,"a":3}
    process_formula = env.translate_formula

if "Letter" in args.env:
    str_to_action = {"w":0,"s":1,"a":2,"d":3}
    process_formula = lambda formula : formula

# set formula
env.sampler.sampled_tasks = args.formula_id


# TEST

obs = env.reset()
done = False
step = 0

while not done:

    step += 1
    env.render()

    print(f"\n---")
    print(f"Step: {step}")
    print(f"Predicted Symbol: {process_formula(env.env.get_events())}")
    print(f"Task:")
    utils.pprint_ltl_formula(process_formula(obs['text']))

    print("\nAction: ", end="")

    if args.input_type == "terminal":
        a = input()
        while a not in str_to_action:
            print("invalid action...")
            print("Action: ", end="")
            a = input()

    elif args.input_type == "keyboard":
        a = None
        while a is None:
            key = cv2.waitKey(100)
            if key == 81 or key == ord('a'):
                a = "a"
            elif key == 82 or key == ord('w'):
                a = "w"
            elif key == 83 or key == ord('d'):
                a = "d"
            elif key == 84 or key == ord('s'):
                a = "s"
        print(a)

    a = str_to_action[a]
    obs, reward, done, info = env.step(a)

    if done:
        env.show()
        print(f"Reward: {reward}")
        print("Done!")
        print("Closing...")
        break

    print(f"Reward: {reward}")

if args.input_type == "terminal":
    input()
elif args.input_type == "keyboard":
    cv2.waitKey(0)

env.close()