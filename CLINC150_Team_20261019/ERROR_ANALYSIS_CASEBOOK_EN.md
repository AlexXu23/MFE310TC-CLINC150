# Error-analysis casebook

All 72 cases are traceable to saved seed-42 predictions. These are AI-assisted interpretations, not independently human-validated annotations. The 20/20/20 sampling allocation is deliberately balanced and cannot estimate corpus category prevalence. No causal feature attribution is claimed.

Utterance source: CLINC150, Stefan Larson et al. (2019), https://aclanthology.org/D19-1131/ ; repository https://github.com/clinc/oos-eval ; repository license CC-BY-3.0, https://github.com/clinc/oos-eval/blob/master/LICENSE . Typo inputs are explicitly modified versions of the original texts. Interpretation notes are new AI-assisted analysis.

## Sampling and coding

Three disjoint raw-MSP/max-validation-OOS-F1 error pools, 20 cases each; 12 extra cases correctly classified clean but misclassified under medium typos. Within each pool, select the first SHA256 keys for `20261019:pool:row_id`. See `outputs/error_analysis/protocol.json`.

## accepted_intent_error (20 cases)

### accepted_intent_error:903

- Original: what's the mileage to get the the taj mahal in atlantic city
- Evaluated: what's the mileage to get the the taj mahal in atlantic city
- Gold → BERT: `distance` → `mpg`; baseline: `mpg`.
- Category: `lexical_cue_ambiguity`. Raw confidence: 0.4796.
- Interpretation: Mileage refers to travel distance to a named place, not fuel efficiency. The distance/mpg distinction depends on the object of the question.
- Human confirmation: pending.

### accepted_intent_error:562

- Original: please switch my name
- Evaluated: please switch my name
- Gold → BERT: `change_user_name` → `change_ai_name`; baseline: `user_name`.
- Category: `semantic_overlap`. Raw confidence: 0.2409.
- Interpretation: The speaker asks to change my name, whereas the prediction changes the assistant's name: same action, different owner.
- Human confirmation: pending.

### accepted_intent_error:1474

- Original: cross off grocery shopping from todo list
- Evaluated: cross off grocery shopping from todo list
- Gold → BERT: `todo_list_update` → `shopping_list_update`; baseline: `todo_list_update`.
- Category: `semantic_overlap`. Raw confidence: 0.2123.
- Interpretation: Grocery shopping is an item on a todo list; the named list is not a shopping list. The baseline preserves this distinction.
- Human confirmation: pending.

### accepted_intent_error:4381

- Original: place my order alexa
- Evaluated: place my order alexa
- Gold → BERT: `order` → `order_status`; baseline: `order_status`.
- Category: `semantic_overlap`. Raw confidence: 0.4499.
- Interpretation: Place my order requests a new action, not the status of an existing order. Both models conflate related commerce intents.
- Human confirmation: pending.

### accepted_intent_error:295

- Original: i have a revolving store card and defaulted so will my fico score be affected
- Evaluated: i have a revolving store card and defaulted so will my fico score be affected
- Gold → BERT: `improve_credit_score` → `damaged_card`; baseline: `replacement_card_duration`.
- Category: `annotation_boundary`. Raw confidence: 0.3757.
- Interpretation: Defaulting and FICO concern credit-score consequences, not a physically damaged card. The gold improve_credit_score label is itself broader than the exact question, so taxonomy ambiguity should be acknowledged.
- Human confirmation: pending.

### accepted_intent_error:1709

- Original: you have yo go to school on calendar
- Evaluated: you have yo go to school on calendar
- Gold → BERT: `calendar` → `calendar_update`; baseline: `calendar_update`.
- Category: `underspecified`. Raw confidence: 0.4575.
- Interpretation: The malformed yo go phrase and on calendar do not clearly distinguish viewing an entry from creating one; both models predict an update.
- Human confirmation: pending.

### accepted_intent_error:3853

- Original: i think all my bills are paid, but can you double check
- Evaluated: i think all my bills are paid, but can you double check
- Gold → BERT: `bill_balance` → `pay_bill`; baseline: `bill_balance`.
- Category: `semantic_overlap`. Raw confidence: 0.3193.
- Interpretation: Double-checking whether bills are paid asks about balance/status, not an instruction to pay. The baseline is correct.
- Human confirmation: pending.

### accepted_intent_error:938

- Original: what's your designation
- Evaluated: what's your designation
- Gold → BERT: `user_name` → `what_is_your_name`; baseline: `what_is_your_name`.
- Category: `annotation_boundary`. Raw confidence: 0.2725.
- Interpretation: Exact text occurs in training as what_is_your_name but in test as user_name. Both models follow the training label; this is a verified cross-split label conflict, not unambiguous model failure.
- Human confirmation: pending.

### accepted_intent_error:1706

- Original: retrieve events on my calendar
- Evaluated: retrieve events on my calendar
- Gold → BERT: `calendar` → `calendar_update`; baseline: `calendar_update`.
- Category: `semantic_overlap`. Raw confidence: 0.3618.
- Interpretation: Retrieve events is a calendar read operation, not a calendar update. Both models confuse the operation boundary.
- Human confirmation: pending.

### accepted_intent_error:2226

- Original: have i told you to add washing dishes to my todo list
- Evaluated: have i told you to add washing dishes to my todo list
- Gold → BERT: `todo_list` → `todo_list_update`; baseline: `todo_list_update`.
- Category: `semantic_overlap`. Raw confidence: 0.4842.
- Interpretation: Have I told you to add is a retrospective query about list contents, not a fresh add instruction. Embedded action wording differs from the outer speech act.
- Human confirmation: pending.

### accepted_intent_error:4378

- Original: what is my payment going to be
- Evaluated: what is my payment going to be
- Gold → BERT: `income` → `min_payment`; baseline: `income`.
- Category: `underspecified`. Raw confidence: 0.2431.
- Interpretation: My payment could mean incoming income or an outgoing minimum payment. The isolated question lacks the account or payer context needed to disambiguate.
- Human confirmation: pending.

### accepted_intent_error:2071

- Original: don't forget to tell me to call my mother
- Evaluated: don't forget to tell me to call my mother
- Gold → BERT: `reminder_update` → `make_call`; baseline: `make_call`.
- Category: `semantic_overlap`. Raw confidence: 0.6877.
- Interpretation: Don't forget to tell me frames a reminder; call my mother is the reminder content, not an immediate call command. Both models predict the embedded action.
- Human confirmation: pending.

### accepted_intent_error:1842

- Original: how much pto is gone
- Evaluated: how much pto is gone
- Gold → BERT: `pto_used` → `pto_balance`; baseline: `pto_used`.
- Category: `semantic_overlap`. Raw confidence: 0.4452.
- Interpretation: PTO gone refers to time already used, whereas pto_balance concerns time remaining. The baseline gets the state distinction right.
- Human confirmation: pending.

### accepted_intent_error:1694

- Original: i can't remember if i added the fun run to my calendar yet; did i
- Evaluated: i can't remember if i added the fun run to my calendar yet; did i
- Gold → BERT: `calendar` → `calendar_update`; baseline: `calendar_update`.
- Category: `semantic_overlap`. Raw confidence: 0.4274.
- Interpretation: Did I add an event asks to inspect an existing calendar state, rather than modify it. Both models predict the embedded update action.
- Human confirmation: pending.

### accepted_intent_error:1804

- Original: can i bring perfume on the plane or what is the size limit
- Evaluated: can i bring perfume on the plane or what is the size limit
- Gold → BERT: `carry_on` → `credit_limit`; baseline: `credit_limit`.
- Category: `lexical_cue_ambiguity`. Raw confidence: 0.2112.
- Interpretation: The size limit is for perfume carried onto a plane, not a financial credit limit. The shared limit wording is an observable ambiguity, not proof of a particular attention mechanism.
- Human confirmation: pending.

### accepted_intent_error:3801

- Original: is garlic on my grocery list
- Evaluated: is garlic on my grocery list
- Gold → BERT: `shopping_list` → `shopping_list_update`; baseline: `todo_list`.
- Category: `semantic_overlap`. Raw confidence: 0.2294.
- Interpretation: Is garlic on my grocery list is membership checking, not adding an item. The baseline also fails, but with a different list type.
- Human confirmation: pending.

### accepted_intent_error:2780

- Original: is mr joes pizza available for a seating
- Evaluated: is mr joes pizza available for a seating
- Gold → BERT: `how_busy` → `restaurant_reservation`; baseline: `meal_suggestion`.
- Category: `semantic_overlap`. Raw confidence: 0.2302.
- Interpretation: Availability for seating can be interpreted as busyness or a reservation request. This is a close service boundary, not an unrelated prediction.
- Human confirmation: pending.

### accepted_intent_error:3864

- Original: how much do i have to pay this month
- Evaluated: how much do i have to pay this month
- Gold → BERT: `bill_balance` → `income`; baseline: `bill_balance`.
- Category: `underspecified`. Raw confidence: 0.2039.
- Interpretation: How much to pay this month lacks an explicit bill/account mention. The gold bill_balance is plausible, but the model predicts incoming income instead.
- Human confirmation: pending.

### accepted_intent_error:3785

- Original: do i have twinkies on the grocery list
- Evaluated: do i have twinkies on the grocery list
- Gold → BERT: `shopping_list` → `shopping_list_update`; baseline: `todo_list`.
- Category: `semantic_overlap`. Raw confidence: 0.2229.
- Interpretation: Do I have an item on the list requests a lookup, not a shopping-list update. The baseline instead confuses the list type.
- Human confirmation: pending.

### accepted_intent_error:3935

- Original: get me my slow sounds
- Evaluated: get me my slow sounds
- Gold → BERT: `play_music` → `change_volume`; baseline: `change_speed`.
- Category: `surface_form`. Raw confidence: 0.1527.
- Interpretation: Slow sounds is an unusual way to request music. The prediction interprets it as an audio-control request; the baseline predicts speed control.
- Human confirmation: pending.

## false_oos (20 cases)

### false_oos:2846

- Original: can you tell me where my item is
- Evaluated: can you tell me where my item is
- Gold → BERT: `order_status` → `order_status`; baseline: `w2`.
- Category: `underspecified`. Raw confidence: 0.0822.
- Interpretation: Where my item is lacks an explicit order or shipment mention. BERT's top intent is nevertheless correct; rejection discards a correct routing decision.
- Human confirmation: pending.

### false_oos:3952

- Original: bump some jams
- Evaluated: bump some jams
- Gold → BERT: `play_music` → `change_volume`; baseline: `travel_suggestion`.
- Category: `surface_form`. Raw confidence: 0.1025.
- Interpretation: Bump some jams is a slang music request. BERT predicts volume control and also rejects it, consistent with weak coverage of this expression.
- Human confirmation: pending.

### false_oos:4308

- Original: what kind of indian would you recommend
- Evaluated: what kind of indian would you recommend
- Gold → BERT: `meal_suggestion` → `translate`; baseline: `what_can_i_ask_you`.
- Category: `underspecified`. Raw confidence: 0.1283.
- Interpretation: What kind of Indian would you recommend omits food or dish, leaving nationality/language readings possible. The model predicts translation and rejects it.
- Human confirmation: pending.

### false_oos:1158

- Original: do i overspend when it comes to fast food
- Evaluated: do i overspend when it comes to fast food
- Gold → BERT: `spending_history` → `spending_history`; baseline: `change_speed`.
- Category: `semantic_overlap`. Raw confidence: 0.1166.
- Interpretation: Overspending on fast food requires a spending-history interpretation. BERT predicts the correct class but rejects this cross-domain food/finance wording.
- Human confirmation: pending.

### false_oos:1404

- Original: can we please
- Evaluated: can we please
- Gold → BERT: `yes` → `maybe`; baseline: `change_language`.
- Category: `underspecified`. Raw confidence: 0.0599.
- Interpretation: Can we please is context-dependent and not a clear standalone yes utterance. The yes/maybe distinction cannot be reliably resolved without dialogue history.
- Human confirmation: pending.

### false_oos:172

- Original: how do i sign up for a new allstatedplan
- Evaluated: how do i sign up for a new allstatedplan
- Gold → BERT: `insurance_change` → `application_status`; baseline: `new_card`.
- Category: `surface_form`. Raw confidence: 0.1113.
- Interpretation: The concatenated allstatedplan and new-plan wording complicate an insurance-change request; the model predicts application status.
- Human confirmation: pending.

### false_oos:4494

- Original: how come i got declined
- Evaluated: how come i got declined
- Gold → BERT: `card_declined` → `card_declined`; baseline: `card_declined`.
- Category: `underspecified`. Raw confidence: 0.1183.
- Interpretation: Got declined omits the card or transaction. BERT predicts card_declined correctly but rejects the underspecified input.
- Human confirmation: pending.

### false_oos:3817

- Original: tell fred that i don't have his guitar
- Evaluated: tell fred that i don't have his guitar
- Gold → BERT: `text` → `share_location`; baseline: `maybe`.
- Category: `semantic_overlap`. Raw confidence: 0.0452.
- Interpretation: Tell Fred requests sending a message, while guitar is its content. The model predicts location sharing, so rejection here accompanies a wrong underlying intent.
- Human confirmation: pending.

### false_oos:1165

- Original: did i stick to my dinner budget
- Evaluated: did i stick to my dinner budget
- Gold → BERT: `spending_history` → `spending_history`; baseline: `meal_suggestion`.
- Category: `semantic_overlap`. Raw confidence: 0.0905.
- Interpretation: Dinner budget combines meal vocabulary with a spending question. BERT chooses spending_history correctly but rejects it; the baseline chooses a meal suggestion.
- Human confirmation: pending.

### false_oos:2409

- Original: yum, pulled pork tacos at the buffet
- Evaluated: yum, pulled pork tacos at the buffet
- Gold → BERT: `restaurant_reviews` → `restaurant_reviews`; baseline: `restaurant_reservation`.
- Category: `annotation_boundary`. Raw confidence: 0.1086.
- Interpretation: The text is an evaluative food statement rather than an explicit request for reviews. BERT predicts the gold restaurant_reviews but rejects it; the label-to-utterance relation merits review.
- Human confirmation: pending.

### false_oos:4489

- Original: i could not buy food using my card when i was in vietnam
- Evaluated: i could not buy food using my card when i was in vietnam
- Gold → BERT: `card_declined` → `card_declined`; baseline: `card_declined`.
- Category: `surface_form`. Raw confidence: 0.1347.
- Interpretation: The narrative description of failing to buy food abroad is correctly classified as card_declined but rejected; the rejection is not an intent-classification error.
- Human confirmation: pending.

### false_oos:3100

- Original: this charge is bs
- Evaluated: this charge is bs
- Gold → BERT: `report_fraud` → `report_fraud`; baseline: `report_fraud`.
- Category: `surface_form`. Raw confidence: 0.1354.
- Interpretation: This charge is bs is an informal dispute/fraud complaint. Both models name report_fraud, but BERT's rejection still loses the supported request.
- Human confirmation: pending.

### false_oos:2337

- Original: i need to know the next billing date of my american express card
- Evaluated: i need to know the next billing date of my american express card
- Gold → BERT: `bill_due` → `replacement_card_duration`; baseline: `expiration_date`.
- Category: `semantic_overlap`. Raw confidence: 0.1022.
- Interpretation: The next billing date is a bill_due question, not replacement-card delivery. The named card product adds information but does not establish a causal entity effect.
- Human confirmation: pending.

### false_oos:4360

- Original: what do i bring in over a year
- Evaluated: what do i bring in over a year
- Gold → BERT: `income` → `what_can_i_ask_you`; baseline: `income`.
- Category: `surface_form`. Raw confidence: 0.0610.
- Interpretation: Bring in over a year is an idiom for annual income. The baseline is correct, while BERT predicts assistant capabilities and rejects.
- Human confirmation: pending.

### false_oos:4353

- Original: what do i take home
- Evaluated: what do i take home
- Gold → BERT: `income` → `order`; baseline: `income`.
- Category: `surface_form`. Raw confidence: 0.0368.
- Interpretation: Take home is an idiom for net income; BERT instead predicts order. The baseline recovers the income reading.
- Human confirmation: pending.

### false_oos:1310

- Original: i didn't understand you
- Evaluated: i didn't understand you
- Gold → BERT: `repeat` → `maybe`; baseline: `repeat`.
- Category: `semantic_overlap`. Raw confidence: 0.1316.
- Interpretation: Didn't understand you pragmatically asks for repetition rather than uncertainty. The baseline predicts repeat; BERT predicts maybe and rejects.
- Human confirmation: pending.

### false_oos:2855

- Original: i would like to change the name on my checking account
- Evaluated: i would like to change the name on my checking account
- Gold → BERT: `pin_change` → `change_ai_name`; baseline: `change_ai_name`.
- Category: `annotation_boundary`. Raw confidence: 0.0928.
- Interpretation: Changing the name on a checking account does not describe changing a PIN. Treat this as a candidate annotation/taxonomy mismatch, not a confirmed relabeling.
- Human confirmation: pending.

### false_oos:391

- Original: can credit card companies expedite delivery of reissued credit cards
- Evaluated: can credit card companies expedite delivery of reissued credit cards
- Gold → BERT: `replacement_card_duration` → `new_card`; baseline: `new_card`.
- Category: `semantic_overlap`. Raw confidence: 0.1347.
- Interpretation: Expediting delivery of a reissued card concerns replacement_card_duration, while both models predict the related new_card intent.
- Human confirmation: pending.

### false_oos:3987

- Original: what can i do to get rid of this event
- Evaluated: what can i do to get rid of this event
- Gold → BERT: `calendar_update` → `calendar_update`; baseline: `calendar_update`.
- Category: `surface_form`. Raw confidence: 0.0925.
- Interpretation: Get rid of this event is an indirect deletion request. BERT predicts calendar_update correctly but rejects it; no literal update verb is required by the intended meaning.
- Human confirmation: pending.

### false_oos:2418

- Original: do people say per se is any good or not
- Evaluated: do people say per se is any good or not
- Gold → BERT: `restaurant_reviews` → `no`; baseline: `restaurant_reviews`.
- Category: `surface_form`. Raw confidence: 0.1291.
- Interpretation: Per Se is a restaurant name embedded in a colloquial review request. BERT predicts no and rejects, whereas the baseline predicts restaurant_reviews; entity causality is not established.
- Human confirmation: pending.

## missed_oos (20 cases)

### missed_oos:5208

- Original: what's the amusement park like
- Evaluated: what's the amusement park like
- Gold → BERT: `oos` → `travel_suggestion`; baseline: `weather`.
- Category: `unsupported_topic`. Raw confidence: 0.4254.
- Interpretation: The question asks about an amusement park rather than requesting a supported travel suggestion. Related travel semantics do not make it a supported intent.
- Human confirmation: pending.

### missed_oos:5274

- Original: what should you do if your laundry machine is making a high pitched whine
- Evaluated: what should you do if your laundry machine is making a high pitched whine
- Gold → BERT: `oos` → `smart_home`; baseline: `what_are_your_hobbies`.
- Category: `unsupported_action`. Raw confidence: 0.3613.
- Interpretation: Troubleshooting a noisy washing machine is not the same as issuing a supported smart-home control request.
- Human confirmation: pending.

### missed_oos:4895

- Original: when will the government file another antitrust suit against microsoft
- Evaluated: when will the government file another antitrust suit against microsoft
- Gold → BERT: `oos` → `report_fraud`; baseline: `payday`.
- Category: `unsupported_topic`. Raw confidence: 0.1685.
- Interpretation: A government antitrust lawsuit is not the personal fraud-reporting service predicted by BERT.
- Human confirmation: pending.

### missed_oos:5332

- Original: rent the tuxedo the day before
- Evaluated: rent the tuxedo the day before
- Gold → BERT: `oos` → `car_rental`; baseline: `date`.
- Category: `unsupported_action`. Raw confidence: 0.3494.
- Interpretation: Tuxedo rental shares the rental action with car_rental but has an unsupported object. Both action and object must be within service scope.
- Human confirmation: pending.

### missed_oos:5082

- Original: does cutting down the rain forest really add to global warming
- Evaluated: does cutting down the rain forest really add to global warming
- Gold → BERT: `oos` → `weather`; baseline: `weather`.
- Category: `unsupported_topic`. Raw confidence: 0.5920.
- Interpretation: Climate-change causation is different from a supported weather query, despite related environmental vocabulary.
- Human confirmation: pending.

### missed_oos:5185

- Original: what kind of foods are border collies allergic to
- Evaluated: what kind of foods are border collies allergic to
- Gold → BERT: `oos` → `nutrition_info`; baseline: `plug_type`.
- Category: `unsupported_topic`. Raw confidence: 0.3240.
- Interpretation: Dog food allergies require veterinary knowledge, not the assistant's supported human nutrition-information intent.
- Human confirmation: pending.

### missed_oos:5238

- Original: deny incoming phone call
- Evaluated: deny incoming phone call
- Gold → BERT: `oos` → `make_call`; baseline: `make_call`.
- Category: `unsupported_action`. Raw confidence: 0.3638.
- Interpretation: Denying an incoming call differs from initiating a call. The make_call prediction collapses this action boundary.
- Human confirmation: pending.

### missed_oos:4567

- Original: where does jelly come from
- Evaluated: where does jelly come from
- Gold → BERT: `oos` → `ingredients_list`; baseline: `where_are_you_from`.
- Category: `unsupported_topic`. Raw confidence: 0.2116.
- Interpretation: The origin of jelly is not necessarily a request for an ingredients list. This food-domain question lies outside the benchmark's supported action.
- Human confirmation: pending.

### missed_oos:5380

- Original: ignore call
- Evaluated: ignore call
- Gold → BERT: `oos` → `make_call`; baseline: `make_call`.
- Category: `unsupported_action`. Raw confidence: 0.3235.
- Interpretation: Ignore call is an incoming-call handling command, not a make_call instruction. Both models predict the supported neighboring action.
- Human confirmation: pending.

### missed_oos:5237

- Original: accept incoming phone call
- Evaluated: accept incoming phone call
- Gold → BERT: `oos` → `make_call`; baseline: `make_call`.
- Category: `unsupported_action`. Raw confidence: 0.5895.
- Interpretation: Accepting an incoming call and making an outgoing call share an object but differ in the requested action.
- Human confirmation: pending.

### missed_oos:5293

- Original: what's a good stretch to do to limber up
- Evaluated: what's a good stretch to do to limber up
- Gold → BERT: `oos` → `travel_suggestion`; baseline: `travel_suggestion`.
- Category: `unsupported_topic`. Raw confidence: 0.1536.
- Interpretation: A stretching recommendation belongs to an unsupported exercise topic; both models route it to travel_suggestion without an evident correct supported class.
- Human confirmation: pending.

### missed_oos:5036

- Original: create a for-while loop that continues 10 times
- Evaluated: create a for-while loop that continues 10 times
- Gold → BERT: `oos` → `timer`; baseline: `calculator`.
- Category: `unsupported_action`. Raw confidence: 0.1529.
- Interpretation: Writing a loop that runs ten times is a programming request, not setting a timer. The number alone cannot define the task.
- Human confirmation: pending.

### missed_oos:4904

- Original: tell me who aphrodite is
- Evaluated: tell me who aphrodite is
- Gold → BERT: `oos` → `who_made_you`; baseline: `who_made_you`.
- Category: `unsupported_topic`. Raw confidence: 0.1948.
- Interpretation: Identifying Aphrodite is a mythology question, not asking who made the assistant. The identity-style phrasing does not determine the supported subject.
- Human confirmation: pending.

### missed_oos:4966

- Original: tell me something about linkin park
- Evaluated: tell me something about linkin park
- Gold → BERT: `oos` → `fun_fact`; baseline: `fun_fact`.
- Category: `scope_boundary`. Raw confidence: 0.3059.
- Interpretation: Tell me something about a music group resembles a generic fun_fact request. Scope depends on the intended service taxonomy, not merely uncertainty in a 150-class classifier.
- Human confirmation: pending.

### missed_oos:5281

- Original: please clean the car windows
- Evaluated: please clean the car windows
- Gold → BERT: `oos` → `schedule_maintenance`; baseline: `smart_home`.
- Category: `unsupported_action`. Raw confidence: 0.2520.
- Interpretation: Cleaning car windows is different from scheduling maintenance, even though both involve vehicle care.
- Human confirmation: pending.

### missed_oos:5432

- Original: print out the photos i took today
- Evaluated: print out the photos i took today
- Gold → BERT: `oos` → `vaccines`; baseline: `meeting_schedule`.
- Category: `unsupported_action`. Raw confidence: 0.1728.
- Interpretation: Printing photos is unsupported. The vaccines prediction has no clear semantic justification; do not invent a keyword-based explanation from this example.
- Human confirmation: pending.

### missed_oos:4747

- Original: disconnect from wifi immediately please
- Evaluated: disconnect from wifi immediately please
- Gold → BERT: `oos` → `sync_device`; baseline: `sync_device`.
- Category: `unsupported_action`. Raw confidence: 0.4634.
- Interpretation: Disconnecting Wi-Fi is a different device action from syncing a device. Both models collapse the neighboring connectivity operations.
- Human confirmation: pending.

### missed_oos:4744

- Original: check the status of my amazon orders for me
- Evaluated: check the status of my amazon orders for me
- Gold → BERT: `oos` → `order_status`; baseline: `order_status`.
- Category: `scope_boundary`. Raw confidence: 0.5433.
- Interpretation: Checking Amazon order status closely resembles the supported order_status label. The benchmark OOS boundary is not obvious from this sentence; this does not by itself prove the gold label wrong.
- Human confirmation: pending.

### missed_oos:5481

- Original: search information on how to apply foundation
- Evaluated: search information on how to apply foundation
- Gold → BERT: `oos` → `new_card`; baseline: `new_card`.
- Category: `unsupported_topic`. Raw confidence: 0.1658.
- Interpretation: Looking up foundation application is a cosmetics information request. The new_card prediction is unsupported, and no causal feature explanation is justified by the text alone.
- Human confirmation: pending.

### missed_oos:5310

- Original: can you check to see which parking garages are open downtown
- Evaluated: can you check to see which parking garages are open downtown
- Gold → BERT: `oos` → `directions`; baseline: `transactions`.
- Category: `unsupported_action`. Raw confidence: 0.1837.
- Interpretation: Checking whether parking garages are open differs from asking for directions, though the destination context is shared.
- Human confirmation: pending.

## typo_induced (12 cases)

### typo_induced:806

- Original: confirm my reservation for joe at hoss's
- Evaluated: cofirm my rservation for joe at hoss's
- Gold → BERT: `confirm_reservation` → `restaurant_reviews`; baseline: `restaurant_reservation`.
- Category: `typo_surface`. Raw confidence: 0.2398.
- Interpretation: Cofirm and rservation corrupt confirmation/reservation cues; a previously correct BERT prediction becomes restaurant_reviews. The baseline also misses the exact intent.
- Human confirmation: pending.

### typo_induced:3927

- Original: how do i cash in my amex reward points
- Evaluated: how do i cash in my amex rewrad poits
- Gold → BERT: `redeem_rewards` → `direct_deposit`; baseline: `redeem_rewards`.
- Category: `typo_surface`. Raw confidence: 0.1356.
- Interpretation: Rewrad and poits damage reward-point wording; BERT switches to direct_deposit while TF-IDF remains correct, showing model-specific sensitivity on identical text.
- Human confirmation: pending.

### typo_induced:1105

- Original: how many points do i have on my visa
- Evaluated: how many poitns do i have on my visa
- Gold → BERT: `rewards_balance` → `apr`; baseline: `rewards_balance`.
- Category: `typo_surface`. Raw confidence: 0.2283.
- Interpretation: Poitns alters the points cue; BERT switches from a correct rewards-balance prediction to APR, while the baseline remains correct.
- Human confirmation: pending.

### typo_induced:3905

- Original: how do i use my card reward points for my next flight
- Evaluated: how do i use my card reward pints for my next flgiht
- Gold → BERT: `redeem_rewards` → `rewards_balance`; baseline: `redeem_rewards`.
- Category: `typo_surface`. Raw confidence: 0.3663.
- Interpretation: Pints and flgiht preserve card/reward wording but change BERT to the neighboring rewards_balance class; the baseline still predicts redemption.
- Human confirmation: pending.

### typo_induced:3246

- Original: let's take a look at the weather
- Evaluated: let's take a look at the wetaher
- Gold → BERT: `weather` → `restaurant_reviews`; baseline: `restaurant_reservation`.
- Category: `typo_surface`. Raw confidence: 0.1795.
- Interpretation: Wetaher corrupts the main weather cue; both models now choose a restaurant-related intent. A direct tokenization effect requires additional evidence, not this example alone.
- Human confirmation: pending.

### typo_induced:830

- Original: put a freeze on my account
- Evaluated: put a freze on my acocunt
- Gold → BERT: `freeze_account` → `schedule_maintenance`; baseline: `freeze_account`.
- Category: `typo_surface`. Raw confidence: 0.0770.
- Interpretation: Freze and acocunt corrupt both action and object. BERT changes to schedule_maintenance but the baseline retains freeze_account.
- Human confirmation: pending.

### typo_induced:3781

- Original: can you check to see if my shopping list already has milk
- Evaluated: can you chck to see if my sopping list already has milk
- Gold → BERT: `shopping_list` → `todo_list`; baseline: `order_status`.
- Category: `typo_surface`. Raw confidence: 0.2762.
- Interpretation: Chck and sopping damage the lookup and shopping cues. BERT confuses the list type; the baseline also fails with an order-status prediction.
- Human confirmation: pending.

### typo_induced:3443

- Original: do i need a vaccine for the congo
- Evaluated: do i need a vacine for the cogo
- Gold → BERT: `vaccines` → `ingredient_substitution`; baseline: `vaccines`.
- Category: `typo_surface`. Raw confidence: 0.1866.
- Interpretation: Vacine and cogo alter the vaccine cue and country name. BERT moves to ingredient_substitution, while the baseline still predicts vaccines.
- Human confirmation: pending.

### typo_induced:1155

- Original: have i gone over my entertainment budget
- Evaluated: have i gone over my enertainment budgt
- Gold → BERT: `spending_history` → `transactions`; baseline: `rollover_401k`.
- Category: `typo_surface`. Raw confidence: 0.0887.
- Interpretation: Enertainment and budgt damage a spending-category query; BERT predicts transactions instead of spending_history. The baseline also fails.
- Human confirmation: pending.

### typo_induced:1464

- Original: what are the benefits associated with my insurance
- Evaluated: what are the beneifts associated with my insuarnce
- Gold → BERT: `insurance` → `meaning_of_life`; baseline: `restaurant_reviews`.
- Category: `typo_surface`. Raw confidence: 0.0568.
- Interpretation: Beneifts and insuarnce damage insurance-benefit wording; both models produce unrelated predictions. The comparison establishes sensitivity, not its internal mechanism.
- Human confirmation: pending.

### typo_induced:4383

- Original: alexa, buy my grocery list
- Evaluated: alxea, buy my groecry list
- Gold → BERT: `order` → `todo_list_update`; baseline: `order`.
- Category: `typo_surface`. Raw confidence: 0.2084.
- Interpretation: Alxea and groecry alter the assistant name and grocery object; BERT changes from order to todo_list_update while the baseline remains correct.
- Human confirmation: pending.

### typo_induced:2348

- Original: how many calories are in a burger
- Evaluated: how many claories are in a bruger
- Gold → BERT: `calories` → `ingredients_list`; baseline: `measurement_conversion`.
- Category: `typo_surface`. Raw confidence: 0.2199.
- Interpretation: Claories and bruger corrupt both main content words; BERT shifts to ingredients_list and TF-IDF to measurement_conversion.
- Human confirmation: pending.
