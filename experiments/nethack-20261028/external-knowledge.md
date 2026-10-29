# External knowledge (from NetHack Wiki + hands-on play)

## Eating food in MiniHack-Eat

- Walk onto the food (%) square. Then use the 'eat' action.
- After choosing 'eat', the game may ask a confirmation like
  "There is an apple here; eat it? [ynq] (n)". Answer 'yes' to eat.
- If you picked up the food first (action 'pickup'), then 'eat' may ask
  "What do you want to eat? [f or ?*]". Reply with the letter of the food
  item shown in the prompt (e.g. 'f' for the apple). This selects which
  inventory item to eat.
- If you get "You cannot eat that!", you selected the wrong item - reply
  with the food item's letter instead.
- Eating the food wins the episode. Do not waste turns on other actions.

## Responding to NetHack prompts (general)

- READ THE GAME MESSAGE every step. If the game asks a question, respond
  to THAT question with the appropriate letter/word - do not ignore it and
  do not repeat the previous action.
- "[ynq]" prompts: 'y' = yes, 'n' = no, 'q' = quit the action.
- "What do you want to X? [abc or ?*]" prompts: respond with the letter
  of the desired item from the list.
- "It's solid stone": you tried to move into a wall. Choose a different
  direction.
- Do not take off equipment, use #shell, jump, or perform unrelated
  actions - they waste your limited steps. Focus on the goal.
