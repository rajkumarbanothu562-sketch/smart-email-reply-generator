# Smart Email Reply Generator - Test Cases & Comparison Table

> **Model**: DistilGPT2 | **Framework**: PyTorch / TorchServe / SageMaker
> **Total Cases**: 10 (5 Formal, 5 Friendly)

| ID | Email Situation | Tone | Generated Reply (Draft) | Latency (ms) |
|---|---|---|---|---|
| **F1** | Requesting leave for one day due to a family medical appointment. | `Formal` | Dear _____________ ,<br>Please take this opportunity by posting an e-mail address in order that we can discuss it with other people who have had problems or may experience any kind (e .g.<br><br>Best regards,<br>[Your Name] | 1408.56 |
| **F2** | Requesting assignment extension for the Machine Learning final project due to illness. | `Formal` | Dear ______________<br>Please note that if there is no opportunity to complete an exam in order not to receive this application please send me another letter explaining how it was arranged by us as well.<br><br>Best regards,<br>[Your Name] | 1196.65 |
| **F3** | Inquiring about the status of a job application for the NLP Engineer role. | `Formal` | Dear ÂCitizen’s representative,<br>Please provide information regarding our office website as well so that we can quickly respond with appropriate response within 24 hours from time-to--we have an opportunity to discuss this issue in detail here .<br><br>Best regards,<br>[Your Name] | 1234.45 |
| **F4** | Declining a vendor sales pitch meeting respectfully due to budget constraints. | `Formal` | Dear  @lindianpapeteles<br><br>Best regards,<br>[Your Name] | 221.07 |
| **F5** | Submitting project status report and requesting executive review. | `Formal` | Dear _______________<br>The following is an excerpt from our presentation with regard thereto :<br>"A survey conducted by Professors [of University of California Berkeley], The American Association For Students in Higher Education (CAUS), has found that students have less than two months left after graduation or they spend most time online searching out their papers.<br><br>Best regards,<br>[Your Name] | 1150.34 |
| **FR1** | Unable to attend tomorrow's meeting because I have an appointment. | `Friendly` | Hi ike,<br>Thank you very much :) Your work is excellent as well. You've been there in my office since last year (thanks again) but it seems that people are just not getting enough attention from their colleagues or other staff at this point. | 1184.76 |
| **FR2** | Thanking a colleague for helping debug the deployment script today. | `Friendly` | Hi !!!<br><br>Cheers,<br>[Your Name] | 1256.07 |
| **FR3** | Inviting teammates for a casual coffee break and project catch-up. | `Friendly` | Hi !!! You're an excellent person...<br><br>Cheers,<br>[Your Name] | 195.42 |
| **FR4** | Sharing notes from the conference and asking if they want to collaborate. | `Friendly` | Hi !!! You're in this room now but did we need any of these? Did someone ask us why it was important that our emails were sent out during dinner or had something special happen at work tomorrow morning (we don't have an answer yet).<br><br>Cheers,<br>[Your Name] | 974.42 |
| **FR5** | Congratulating a teammate on their work anniversary and promotion. | `Friendly` | Hi !!! You're my friend; how can we do it? It's an easy task...and one that is always hard or difficult - but sometimes even harder than others in our company....to find out who gets paid by other companies when they don't pay attention to themselves first-time customers at all (because of this!).<br><br>Cheers,<br>[Your Name] | 1270.45 |


## Detailed Test Case Audit

### Case F1: Requesting leave for one day due to a family medical appointment. (Formal)

- **Expected Behavior**: Generate a polite, professional leave-related draft.
- **Constructed Prompt**:
```text
Task: Generate a polite and professional formal email reply based on the situation.

Example 1:
Situation: Requesting leave for one day
Tone: Formal
Draft Reply:
Dear Manager,

I am writing to formally request one day of leave on Friday due to personal matters. I will ensure all pressing tasks are completed before my departure.

Sincerely,
[Your Name]

Example 2:
Situation: Unable to attend quarterly review meeting
Tone: Formal
Draft Reply:
Dear Colleagues,

Thank you for the meeting invitation. Regrettably, I will be unable to attend the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.

Best regards,
[Your Name]

Situation: Requesting leave for one day due to a family medical appointment.
Tone: Formal
Draft Reply:
Dear 
```
- **Generated Draft Reply**:
```text
Dear _____________ ,
Please take this opportunity by posting an e-mail address in order that we can discuss it with other people who have had problems or may experience any kind (e .g.

Best regards,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case F2: Requesting assignment extension for the Machine Learning final project due to illness. (Formal)

- **Expected Behavior**: Generate a respectful academic email draft.
- **Constructed Prompt**:
```text
Task: Generate a polite and professional formal email reply based on the situation.

Example 1:
Situation: Requesting leave for one day
Tone: Formal
Draft Reply:
Dear Manager,

I am writing to formally request one day of leave on Friday due to personal matters. I will ensure all pressing tasks are completed before my departure.

Sincerely,
[Your Name]

Example 2:
Situation: Unable to attend quarterly review meeting
Tone: Formal
Draft Reply:
Dear Colleagues,

Thank you for the meeting invitation. Regrettably, I will be unable to attend the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.

Best regards,
[Your Name]

Situation: Requesting assignment extension for the Machine Learning final project due to illness.
Tone: Formal
Draft Reply:
Dear 
```
- **Generated Draft Reply**:
```text
Dear ______________
Please note that if there is no opportunity to complete an exam in order not to receive this application please send me another letter explaining how it was arranged by us as well.

Best regards,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case F3: Inquiring about the status of a job application for the NLP Engineer role. (Formal)

- **Expected Behavior**: Generate a courteous professional status inquiry draft.
- **Constructed Prompt**:
```text
Task: Generate a polite and professional formal email reply based on the situation.

Example 1:
Situation: Requesting leave for one day
Tone: Formal
Draft Reply:
Dear Manager,

I am writing to formally request one day of leave on Friday due to personal matters. I will ensure all pressing tasks are completed before my departure.

Sincerely,
[Your Name]

Example 2:
Situation: Unable to attend quarterly review meeting
Tone: Formal
Draft Reply:
Dear Colleagues,

Thank you for the meeting invitation. Regrettably, I will be unable to attend the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.

Best regards,
[Your Name]

Situation: Inquiring about the status of a job application for the NLP Engineer role.
Tone: Formal
Draft Reply:
Dear 
```
- **Generated Draft Reply**:
```text
Dear ÂCitizen’s representative,
Please provide information regarding our office website as well so that we can quickly respond with appropriate response within 24 hours from time-to--we have an opportunity to discuss this issue in detail here .

Best regards,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case F4: Declining a vendor sales pitch meeting respectfully due to budget constraints. (Formal)

- **Expected Behavior**: Generate a polite, formal declination email draft.
- **Constructed Prompt**:
```text
Task: Generate a polite and professional formal email reply based on the situation.

Example 1:
Situation: Requesting leave for one day
Tone: Formal
Draft Reply:
Dear Manager,

I am writing to formally request one day of leave on Friday due to personal matters. I will ensure all pressing tasks are completed before my departure.

Sincerely,
[Your Name]

Example 2:
Situation: Unable to attend quarterly review meeting
Tone: Formal
Draft Reply:
Dear Colleagues,

Thank you for the meeting invitation. Regrettably, I will be unable to attend the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.

Best regards,
[Your Name]

Situation: Declining a vendor sales pitch meeting respectfully due to budget constraints.
Tone: Formal
Draft Reply:
Dear 
```
- **Generated Draft Reply**:
```text
Dear  @lindianpapeteles

Best regards,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case F5: Submitting project status report and requesting executive review. (Formal)

- **Expected Behavior**: Generate a formal deliverable submission email draft.
- **Constructed Prompt**:
```text
Task: Generate a polite and professional formal email reply based on the situation.

Example 1:
Situation: Requesting leave for one day
Tone: Formal
Draft Reply:
Dear Manager,

I am writing to formally request one day of leave on Friday due to personal matters. I will ensure all pressing tasks are completed before my departure.

Sincerely,
[Your Name]

Example 2:
Situation: Unable to attend quarterly review meeting
Tone: Formal
Draft Reply:
Dear Colleagues,

Thank you for the meeting invitation. Regrettably, I will be unable to attend the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.

Best regards,
[Your Name]

Situation: Submitting project status report and requesting executive review.
Tone: Formal
Draft Reply:
Dear 
```
- **Generated Draft Reply**:
```text
Dear _______________
The following is an excerpt from our presentation with regard thereto :
"A survey conducted by Professors [of University of California Berkeley], The American Association For Students in Higher Education (CAUS), has found that students have less than two months left after graduation or they spend most time online searching out their papers.

Best regards,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case FR1: Unable to attend tomorrow's meeting because I have an appointment. (Friendly)

- **Expected Behavior**: Generate a courteous, conversational message.
- **Constructed Prompt**:
```text
Task: Generate a warm and friendly email reply based on the situation.

Example 1:
Situation: Thanking a colleague for help
Tone: Friendly
Draft Reply:
Hi Alex,

Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.

Cheers,
[Your Name]

Example 2:
Situation: Grabbing lunch next Tuesday
Tone: Friendly
Draft Reply:
Hey Chris,

Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.

Talk soon,
[Your Name]

Situation: Unable to attend tomorrow's meeting because I have an appointment.
Tone: Friendly
Draft Reply:
Hi 
```
- **Generated Draft Reply**:
```text
Hi ike,
Thank you very much :) Your work is excellent as well. You've been there in my office since last year (thanks again) but it seems that people are just not getting enough attention from their colleagues or other staff at this point.
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case FR2: Thanking a colleague for helping debug the deployment script today. (Friendly)

- **Expected Behavior**: Generate a brief appreciative reply.
- **Constructed Prompt**:
```text
Task: Generate a warm and friendly email reply based on the situation.

Example 1:
Situation: Thanking a colleague for help
Tone: Friendly
Draft Reply:
Hi Alex,

Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.

Cheers,
[Your Name]

Example 2:
Situation: Grabbing lunch next Tuesday
Tone: Friendly
Draft Reply:
Hey Chris,

Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.

Talk soon,
[Your Name]

Situation: Thanking a colleague for helping debug the deployment script today.
Tone: Friendly
Draft Reply:
Hi 
```
- **Generated Draft Reply**:
```text
Hi !!!

Cheers,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case FR3: Inviting teammates for a casual coffee break and project catch-up. (Friendly)

- **Expected Behavior**: Generate an informal, cheerful invitation draft.
- **Constructed Prompt**:
```text
Task: Generate a warm and friendly email reply based on the situation.

Example 1:
Situation: Thanking a colleague for help
Tone: Friendly
Draft Reply:
Hi Alex,

Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.

Cheers,
[Your Name]

Example 2:
Situation: Grabbing lunch next Tuesday
Tone: Friendly
Draft Reply:
Hey Chris,

Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.

Talk soon,
[Your Name]

Situation: Inviting teammates for a casual coffee break and project catch-up.
Tone: Friendly
Draft Reply:
Hi 
```
- **Generated Draft Reply**:
```text
Hi !!! You're an excellent person...

Cheers,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case FR4: Sharing notes from the conference and asking if they want to collaborate. (Friendly)

- **Expected Behavior**: Generate an enthusiastic, collaborative peer draft.
- **Constructed Prompt**:
```text
Task: Generate a warm and friendly email reply based on the situation.

Example 1:
Situation: Thanking a colleague for help
Tone: Friendly
Draft Reply:
Hi Alex,

Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.

Cheers,
[Your Name]

Example 2:
Situation: Grabbing lunch next Tuesday
Tone: Friendly
Draft Reply:
Hey Chris,

Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.

Talk soon,
[Your Name]

Situation: Sharing notes from the conference and asking if they want to collaborate.
Tone: Friendly
Draft Reply:
Hi 
```
- **Generated Draft Reply**:
```text
Hi !!! You're in this room now but did we need any of these? Did someone ask us why it was important that our emails were sent out during dinner or had something special happen at work tomorrow morning (we don't have an answer yet).

Cheers,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

### Case FR5: Congratulating a teammate on their work anniversary and promotion. (Friendly)

- **Expected Behavior**: Generate a warm, congratulatory email draft.
- **Constructed Prompt**:
```text
Task: Generate a warm and friendly email reply based on the situation.

Example 1:
Situation: Thanking a colleague for help
Tone: Friendly
Draft Reply:
Hi Alex,

Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.

Cheers,
[Your Name]

Example 2:
Situation: Grabbing lunch next Tuesday
Tone: Friendly
Draft Reply:
Hey Chris,

Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.

Talk soon,
[Your Name]

Situation: Congratulating a teammate on their work anniversary and promotion.
Tone: Friendly
Draft Reply:
Hi 
```
- **Generated Draft Reply**:
```text
Hi !!! You're my friend; how can we do it? It's an easy task...and one that is always hard or difficult - but sometimes even harder than others in our company....to find out who gets paid by other companies when they don't pay attention to themselves first-time customers at all (because of this!).

Cheers,
[Your Name]
```
- **Review Status**: Draft (Requires Review) (Not sent automatically)

---

