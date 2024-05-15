import datetime
import os
import random
import time
import copy
import json
import glob
import importlib
import numpy as np
import torch
import transformers

import sys

import alfworld.agents.environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDQNAgent
from alfworld.agents.eval import evaluate_dqn
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory
from alfworld.agents.utils.misc import extract_admissible_commands

from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def llm_call(pipeline, prev_rewards, most_recent_obs_list, task_desc, obs_list):

    prompts = []
    for most_recent_obs, obs in zip(most_recent_obs_list, obs_list):

        prompt = f"Task Description: {task_desc}\nObservations: {most_recent_obs}\nCurrent State: {obs}\n\nScore the current game progress on a scale from 0 to 3. Only output the float value of the score."
        messages=[
                {"role": "system", "content": "You are guiding an agent to complete a task. The input provided to you will include:\n\t1.\tTask Description”, which outlines the objective the agent needs to achieve. \n\t2.\tObservations: which consists of a series of states achieved by the agent and the corresponding outcomes, described step-by-step. The state are separated by [SEP]. You must record each action’s effect on the environment, allowing you to track how the agent is progressing with the task.\n\t3.\t“Current”, which tells you the current environment perceived by the agent. \nYou will guide the agent by outputting only a numerical score ranging from 0 to 3. This reward will help the agent know its progress toward completing the task. A score of 0 tells the agent that it has made no progress. This example should be scored 0: \nTask Description: put a clean cloth in drawer.\nObservations: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a countertop 1, a drawer 1, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, a towelholder 1, and a towelholder 2. \nStates: [SEP] You arrive at loc 3. On the garbagecan 1, you see nothing. [SEP] go to garbagecan 1 [SEP] You arrive at loc 7. On the toiletpaperhanger 1, you see a toiletpaper 1. [SEP] go to toiletpaperhanger 1 [SEP] You arrive at loc 19. On the bathtubbasin 1, you see nothing. [SEP] go to bathtubbasin 1\nCurrent State: You arrive at loc 22. On the towelholder 1, you see a towel 1.  This is scored zero because the agent has not yet discovered a clean clot. \nA score of 1 tells the agent that it has made intermediate progress on the task. This example should be scored 1: \nTask Description: cool some pan and put it in diningtable.\nObservations: You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 10, a cabinet 11, a cabinet 2, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a diningtable 1, a drawer 1, a drawer 2, a drawer 3, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, and a toaster 1. \nStates: [SEP] You arrive at loc 32. On the stoveburner 4, you see a kettle 1, and a pot 1. [SEP] go to stoveburner 4 [SEP] On the stoveburner 4, you see a kettle 1, and a pot 1. [SEP] examine stoveburner 4 [SEP] You arrive at loc 29. On the stoveburner 3, you see a pan 1. [SEP] go to stoveburner 3\nCurrent State: On the stoveburner 3, you see a pan 1.\nBecause the agent has found the pan, but has neither cooled it nor found the dining table.\nA score of 2 tells the agent that it has made strong progress on the task. This example should be scored 2:\nTask Description: put some soapbottle on cart.\nObservations: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cart 1, a countertop 1, a drawer 1, a drawer 2, a drawer 3, a drawer 4, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1. \nObservations: [SEP] You arrive at loc 17. On the sinkbasin 1, you see a soapbar 2. [SEP] go to sinkbasin 1 [SEP] You arrive at loc 20. On the shelf 1, you see a candle 1, a cloth 1, a soapbar 1, a soapbottle 1, a soapbottle 2, and a tissuebox 1. [SEP] go to shelf 1 [SEP] You arrive at loc 8. On the toilet 1, you see a spraybottle 1, and a spraybottle 2. [SEP] go to toilet 1\nCurrent State: You arrive at loc 21. On the countertop 1, you see a dishsponge 1, and a soapbottle 3.\nBecause the agent has found the soap bottle and will shortly pick it up, but still needs to do the second part of the task, which is placing it on the cart.\nA score of 3 tells the agent that it has nearly completed the task. This example should be scored 3: \nTask Description: look at pen under the desklamp.\nObservations: You are in the middle of a room. Looking quickly around you, you see a armchair 1, a armchair 2, a armchair 3, a bed 1, a cabinet 1, a cabinet 2, a desk 1, a drawer 1, a drawer 2, a drawer 3, a drawer 4, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1. \nStates: [SEP] You arrive at loc 29. On the sidetable 1, you see a alarmclock 1, a desklamp 1, a keychain 1, and a pen 1. [SEP] go to sidetable 1 [SEP] You pick up the pen 1 from the sidetable 1. [SEP] take pen 1 from sidetable 1 [SEP] You arrive at loc 25. On the safe 1, you see nothing. [SEP] go to safe 1\nCurrent State: You arrive at loc 29. On the sidetable 1, you see a alarmclock 1, a desklamp 1, and a keychain 1.\nThis is because the agent has found a pen, picked it up, and found a desk lamp.\nYour numerical output between 0 and 3 will be very helpful for the agent to learn how to navigate its environment and accomplish the task specified in “Task Description”"},
                {"role": "user", "content": prompt}
            ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        prompts.append(prompt)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompts,
        max_new_tokens=5,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    scores = []
    for output, prompt in zip(outputs, prompts):
        score = output[0]["generated_text"][len(prompt):]
        scores.append(float(score))
    return scores


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    agent = TextDQNAgent(config)

    env_type = config["env"]["type"]
    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    if env_type == "Hybrid":
        thor = getattr(alfworld.agents.environment, "AlfredThorEnv")(config)
        tw = getattr(alfworld.agents.environment, "AlfredTWEnv")(config)

        thor_env = thor.init_env(batch_size=agent.batch_size)
        tw_env = tw.init_env(batch_size=agent.batch_size)
    else:
        alfred_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="train")
        env = alfred_env.init_env(batch_size=agent.batch_size)

        if agent.run_eval:
            # in distribution
            if config['dataset']['eval_id_data_path'] is not None:
                alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_in_distribution")
                id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
                num_id_eval_game = alfred_env.num_games
            # out of distribution
            if config['dataset']['eval_ood_data_path'] is not None:
                alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
                ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
                num_ood_eval_game = alfred_env.num_games

    output_dir = config["general"]["save_path"]
    data_dir = config["general"]["save_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        reward_win, step_win = None, None
        dqn_loss_win = None
        viz_game_points, viz_step, viz_overall_rewards = [], [], []
        viz_id_eval_game_points, viz_id_eval_step = [], []
        viz_ood_eval_game_points, viz_ood_eval_step = [], []
        viz_dqn_loss = []

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_overall_rewards = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_dqn_loss = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far, best_ood_performance_so_far = 0.0, 0.0
    episodic_counting_memory = EpisodicCountingMemory()  # episodic counting based memory
    obj_centric_episodic_counting_memory = ObjCentricEpisodicMemory()

    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break

        # hybrid env switching
        if env_type == "Hybrid":
            if random.uniform(0, 1) < config["env"]["hybrid_tw_prob"]:
                env = tw_env
            else:
                env = thor_env

        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)
        episodic_counting_memory.reset()  # reset episodic counting based memory
        obj_centric_episodic_counting_memory.reset() # reset object centric episodic counting based memory
        previous_dynamics = None

        chosen_actions = []
        prev_step_dones, prev_rewards = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)

        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = copy.deepcopy(observation_strings)
        agent.observation_pool.push_first_sight(first_sight_strings)
        if agent.action_space == "exhaustive":
            action_candidate_list = [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
        else:
            action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
        observation_only = observation_strings
        observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, chosen_actions)]  # appending the chosen action at previous step into the observation
        episodic_counting_memory.push(observation_only)  # update init observation into memory
        obj_centric_episodic_counting_memory.push(observation_only)

        # it requires to store sequences of transitions into memory with order,
        # so we use a cache to keep what agents returns, and push them into memory
        # altogether in the end of game.
        transition_cache = []
        still_running_mask = []
        sequence_game_rewards, sequence_count_rewards, sequence_novel_object_rewards, sequence_game_points = [], [], [], []
        print_actions = []

        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode
        for step_no in range(agent.max_nb_steps_per_episode):
            # push obs into observation pool
            agent.observation_pool.push_batch(observation_strings)
            # get most recent k observations
            most_recent_observation_strings = agent.observation_pool.get()
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights

            # predict actions
            if agent.action_space == "generation":
                chosen_actions, chosen_indices, current_dynamics = agent.command_generation_act(most_recent_observation_strings, task_desc_strings, previous_dynamics, random=act_randomly)
            elif agent.action_space == "beam_search_choice":
                chosen_actions, chosen_indices, current_dynamics, action_candidate_list = agent.beam_search_choice_act(most_recent_observation_strings, task_desc_strings, previous_dynamics, random=act_randomly)
            elif agent.action_space in ["admissible", "exhaustive"]:
                chosen_actions, chosen_indices, current_dynamics = agent.admissible_commands_act(most_recent_observation_strings, task_desc_strings, action_candidate_list, previous_dynamics, random=act_randomly)
            else:
                raise NotImplementedError()

            replay_info = [most_recent_observation_strings, task_desc_strings, action_candidate_list, chosen_indices]
            transition_cache.append(replay_info)
            obs, _, dones, infos = env.step(chosen_actions)
            #scores = [float(item) for item in infos["won"]]

            scores = llm_call(pipeline, prev_rewards,most_recent_observation_strings,task_desc_strings,obs)
            #scores = [0] * len(dones) 
            #for i in range(len(scores)):
                #scores[i] = llm_call(pipeline, prev_rewards[i],most_recent_observation_strings[i],task_desc_strings[i],obs[i])

            dones = [float(item) for item in dones]

            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            if agent.action_space == "exhaustive":
                action_candidate_list = [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
            else:
                action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
            observation_only = observation_strings
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, chosen_actions)]  # appending the chosen action at previous step into the observation
            seeing_new_states = episodic_counting_memory.is_a_new_state(observation_only)
            seeing_new_objects = obj_centric_episodic_counting_memory.get_object_novelty_reward(observation_only)
            episodic_counting_memory.push(observation_only)  # update new observation into memory
            obj_centric_episodic_counting_memory.push(observation_only)  # update new observation into memory
            previous_dynamics = current_dynamics

            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                dqn_loss, _ = agent.update_dqn()
                if dqn_loss is not None:
                    running_avg_dqn_loss.push(dqn_loss)

            if step_no == agent.max_nb_steps_per_episode - 1:
                # terminate the game because DQN requires one extra step
                dones = [1.0 for _ in dones]

            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            count_rewards = [r * agent.count_reward_lambda for r in seeing_new_states]  # list of float
            novel_object_rewards = [r * agent.novel_object_reward_lambda for r in seeing_new_objects] # list of novel object rewards
            sequence_game_points.append(copy.copy(step_rewards))
            prev_rewards = scores
            still_running_mask.append(still_running)
            sequence_game_rewards.append(step_rewards)
            sequence_count_rewards.append(count_rewards)
            sequence_novel_object_rewards.append(novel_object_rewards)
            print_actions.append(chosen_actions[0] if still_running[0] else "--")

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        still_running_mask_np = np.array(still_running_mask)
        game_rewards_np = np.array(sequence_game_rewards) * still_running_mask_np  # step x batch
        count_rewards_np = np.array(sequence_count_rewards) * still_running_mask_np  # step x batch
        novel_object_rewards_np = np.array(sequence_novel_object_rewards) * still_running_mask_np
        game_points_np = np.array(sequence_game_points) * still_running_mask_np  # step x batch
        game_rewards_pt = generic.to_pt(game_rewards_np, enable_cuda=False, type='float')  # step x batch
        count_rewards_pt = generic.to_pt(count_rewards_np, enable_cuda=False, type='float')  # step x batch
        novel_object_rewards_pt = generic.to_pt(novel_object_rewards_np, enable_cuda=False, type='float')

        # push experience into replay buffer (dqn)
        avg_reward_in_replay_buffer = agent.dqn_memory.get_avg_rewards()
        for b in range(game_rewards_np.shape[1]):
            if still_running_mask_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                avg_reward = game_rewards_np[:, b].tolist() + [0.0]
                _need_pad = True
            else:
                avg_reward = game_rewards_np[:, b]
                _need_pad = False
            avg_reward = np.mean(avg_reward)
            is_prior = avg_reward >= avg_reward_in_replay_buffer

            mem = []
            for i in range(game_rewards_np.shape[0]):
                observation_strings, task_strings, action_candidate_list, chosen_indices = transition_cache[i]
                mem.append([observation_strings[b],
                            task_strings[b],
                            action_candidate_list[b],
                            chosen_indices[b],
                            game_rewards_pt[i][b], count_rewards_pt[i][b], novel_object_rewards_pt[i][b]])
                if still_running_mask_np[i][b] == 0.0:
                    break
            if _need_pad:
                observation_strings, task_strings, action_candidate_list, chosen_indices = transition_cache[-1]
                mem.append([observation_strings[b],
                           task_strings[b],
                           action_candidate_list[b],
                           chosen_indices[b],
                           game_rewards_pt[-1][b] * 0.0, count_rewards_pt[-1][b] * 0.0, novel_object_rewards_pt[-1][b] * 0.0])
            agent.dqn_memory.push(is_prior, avg_reward, mem)

        for b in range(batch_size):
            running_avg_game_points.push(np.sum(game_points_np, 0)[b])
            running_avg_overall_rewards.push(np.sum(game_rewards_np, 0)[b] + np.sum(count_rewards_np, 0)[b] + np.sum(novel_object_rewards_np, 0)[b])
            running_avg_game_steps.push(np.sum(still_running_mask_np, 0)[b])

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size
        print("\n"+str(episode_no%batch_size))

        if episode_no < agent.learn_start_from_this_episode:
            continue
        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - batch_size) % agent.report_frequency):
            continue
        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | dqn loss: {:2.3f} | overall rewards: {:2.3f}/{:2.3f} | game points: {:2.3f}/{:2.3f} | used steps: {:2.3f}/{:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_dqn_loss.get_avg(), np.mean(np.sum(game_rewards_np, 0) + np.sum(count_rewards_np, 0) + np.sum(novel_object_rewards_np, 0)), running_avg_overall_rewards.get_avg(), np.mean(np.sum(game_points_np, 0)), running_avg_game_points.get_avg(), np.mean(np.sum(still_running_mask_np, 0)), running_avg_game_steps.get_avg()))
        # print(game_id + ":    " + " | ".join(print_actions))
        print(" | ".join(print_actions))

        # evaluate
        id_eval_game_points, id_eval_game_step = 0.0, 0.0
        ood_eval_game_points, ood_eval_game_step = 0.0, 0.0
        if agent.run_eval:
            if id_eval_env is not None:
                id_eval_res = evaluate_dqn(id_eval_env, agent, num_id_eval_game)
                id_eval_game_points, id_eval_game_step = id_eval_res['average_points'], id_eval_res['average_steps']
            if ood_eval_env is not None:
                ood_eval_res = evaluate_dqn(ood_eval_env, agent, num_ood_eval_game)
                ood_eval_game_points, ood_eval_game_step = ood_eval_res['average_points'], ood_eval_res['average_steps']
            if id_eval_game_points >= best_performance_so_far:
                best_performance_so_far = id_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_id.pt")
            if ood_eval_game_points >= best_ood_performance_so_far:
                best_ood_performance_so_far = ood_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_ood.pt")
        else:
            if running_avg_game_points.get_avg() >= best_performance_so_far:
                best_performance_so_far = running_avg_game_points.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")

        # plot using visdom
        if config["general"]["visdom"]:
            viz_game_points.append(running_avg_game_points.get_avg())
            viz_overall_rewards.append(running_avg_overall_rewards.get_avg())
            viz_step.append(running_avg_game_steps.get_avg())
            viz_dqn_loss.append(running_avg_dqn_loss.get_avg())
            viz_id_eval_game_points.append(id_eval_game_points)
            viz_id_eval_step.append(id_eval_game_step)
            viz_ood_eval_game_points.append(ood_eval_game_points)
            viz_ood_eval_step.append(ood_eval_game_step)
            viz_x = np.arange(len(viz_game_points)).tolist()

            if reward_win is None:
                reward_win = viz.line(X=viz_x, Y=viz_game_points,
                            opts=dict(title=agent.experiment_tag + "_game_points"),
                        name="game points")
                viz.line(X=viz_x, Y=viz_overall_rewards,
                            opts=dict(title=agent.experiment_tag + "_overall_rewards"),
                            win=reward_win, update='append', name="overall rewards")
                viz.line(X=viz_x, Y=viz_id_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                            win=reward_win, update='append', name="id eval game points")
                viz.line(X=viz_x, Y=viz_ood_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win, update='append', name="ood eval game points")
            else:
                viz.line(X=[len(viz_game_points) - 1], Y=[viz_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_game_points"),
                            win=reward_win,
                            update='append', name="game points")
                viz.line(X=[len(viz_overall_rewards) - 1], Y=[viz_overall_rewards[-1]],
                            opts=dict(title=agent.experiment_tag + "_overall_rewards"),
                            win=reward_win,
                            update='append', name="overall rewards")
                viz.line(X=[len(viz_id_eval_game_points) - 1], Y=[viz_id_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                            win=reward_win,
                            update='append', name="id eval game points")
                viz.line(X=[len(viz_ood_eval_game_points) - 1], Y=[viz_ood_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win,
                            update='append', name="ood eval game points")

            if step_win is None:
                step_win = viz.line(X=viz_x, Y=viz_step,
                                    opts=dict(title=agent.experiment_tag + "_step"),
                                    name="step")
                viz.line(X=viz_x, Y=viz_id_eval_step,
                            opts=dict(title=agent.experiment_tag + "_id_eval_step"),
                            win=step_win, update='append', name="id eval step")
                viz.line(X=viz_x, Y=viz_ood_eval_step,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_step"),
                            win=step_win, update='append', name="ood eval step")
            else:
                viz.line(X=[len(viz_step) - 1], Y=[viz_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_step"),
                            win=step_win,
                            update='append', name="step")
                viz.line(X=[len(viz_id_eval_step) - 1], Y=[viz_id_eval_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_id_eval_step"),
                            win=step_win,
                            update='append', name="id eval step")
                viz.line(X=[len(viz_ood_eval_step) - 1], Y=[viz_ood_eval_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_ood_eval_step"),
                            win=step_win,
                            update='append', name="ood eval step")

            if dqn_loss_win is None:
                dqn_loss_win = viz.line(X=viz_x, Y=viz_dqn_loss,
                                    opts=dict(title=agent.experiment_tag + "_dqn_loss"),
                                    name="dqn loss")
            else:
                viz.line(X=[len(viz_dqn_loss) - 1], Y=[viz_dqn_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_dqn_loss"),
                            win=dqn_loss_win,
                            update='append', name="dqn loss")

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                            "dqn loss": str(running_avg_dqn_loss.get_avg()),
                            "overall rewards": str(running_avg_overall_rewards.get_avg()),
                            "train game points": str(running_avg_game_points.get_avg()),
                            "train steps": str(running_avg_game_steps.get_avg()),
                            "id eval game points": str(id_eval_game_points),
                            "id eval steps": str(id_eval_game_step),
                            "ood eval game points": str(ood_eval_game_points),
                            "ood eval steps": str(ood_eval_game_step)})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()
    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_final.pt")


if __name__ == '__main__':
    train()
