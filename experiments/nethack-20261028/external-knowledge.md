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

## Wielding / equipping items (MiniHack-Wield and similar)

- Walk onto the weapon/item symbol (')' is a weapon), then press ',' (pickup).
  The message will show the item's inventory slot, e.g. "f - a dagger".
- Press 'w' (wield). The game asks "What do you want to wield? [- abf or ?*]".
  The letters in brackets are INVENTORY SLOTS: a/b/c... are your starting
  items, later letters (f, g...) are items you picked up this episode.
- Reply with the SLOT LETTER of the weapon you want to wield (e.g. 'f' for
  the just-picked dagger). "You are already wielding that!" means that slot
  is your current weapon - pick a different slot letter.
- Wielding the correct weapon wins the episode.

## General item-prompt rule (applies to wear, puton, read, zap, eat...)

- When the game asks "What do you want to X? [letters or ?*]", the letters
  are INVENTORY SLOTS. Reply with the slot letter of the item you want.
- Pick up an item first (',' on its square) - the message tells you its new
  slot letter. Then use the command and reply with that slot letter.
