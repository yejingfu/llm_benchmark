from openai import OpenAI
import argparse
import time

base_url = "http://localhost:8000/v1"
#api_key = "123456"
api_key = "8d18e8be-9add-4fc5-adb4-8d525a4c2cf3"
# base_url="https://api.deepinfra.com/v1/openai"
# api_key = "xxx"
# base_url="https://api.novita.ai/v3/openai"
# api_key="8d18e8be-xxx-8d525a4c2cf3"

system_prompt = """You are a job post writer. Create ONLY this JSON:
{"titles":"<title>","descriptions":"<description>","lang_code":"<language>"}

REQUIRED sections (use EXACTLY this format):
"Sobre nosotros: [company description in 2 clear sentences]<br><br>"

"Responsabilidades principales:<br>
[clear task 1]<br>
[clear task 2]<br>
[clear task 3]<br><br>"

"Requisitos:<br>
[clear requirement 1]<br>
[clear requirement 2]<br><br>"

"Beneficios:<br>
[clear benefit 1]<br>
[clear benefit 2]"

CRITICAL:
Each sentence MUST be complete
NO partial or nonsense text
NO sudden stops"""


#print(f"===== {system_prompt}")

user_prompts = [
"""Analyze this job posting and create a unique rewritten version in the same language. Return as minified JSON with format {"titles": NEW_TITLE, "descriptions": NEW_DESCRIPTION, "lang_code": LANGUAGE_CODE}. Make it engaging while keeping core requirements: "<h2>Assistant Deli Manager</h2><br><h3> FULL-TIME </h3> <p>$17/hr</p> <p>Responsible for the receiving product and ensures that aisles/freezers and refrigerators are stocked, labeled, clean and delivered product is packed out, and proper customer service is provided. 50% or more of the job is manual labor.</p> <p>Essential Functions:<br><br>- Ensure proper customer service and works to develop relationships with large customers.<br><br>- Supervises and works together with Deli Supervisor (if applicable) and Stocker/s to assure that shelves are stocked and merchandise is rotated.<br><br>- Assists in developing schedules, monitors performance and recommends the proper discipline as appropriate including termination.<br><br>- Trains employees in job responsibilities and safe operating procedures and interviews candidates and recommends for hires.<br><br>- Ensures that employees are performing the proper inspections to meet HACCP regulations as well as conducting periodic HACCP audits.<br><br>- Reviews inventory for product rotation on a daily basis to prevent shrinkage and damages.<br><br>- Ensures that shelf pricing is correct and reflects the most recent pricing and market conditions.<br><br>- Supervises and works alongside the Stocker/s in the receiving of all deli products and ensures that the proper paperwork is completed.<br><br>- Maintains accurate computer inventory levels by having physical inventories performed on a regular basis and adjustments made.<br><br>- Maintains refrigerated equipment and makes sure maintenance contracts and schedules are followed.<br><br>- Supervises the ordering of deli products from vendors on a regular basis to assure we have competitive pricing and minimal shrink due to spoilage and not have too much inventory on hand.<br><br>- Makes sure all the employees in the department can work the equipment such as Toledo scale and Dennison label machine.<br><br>- Coordinates that the pallets stored in the racks have the proper block and date tags.<br><br>- Follows program to maintain the cleanliness of the area by a regular maintenance schedule of scrubbing, and pulling out pallets and cleaning underneath.</p> BOS-04 WS-04 WS-TC <br><b>Schedule</b><br> <p> Shift start: 6:00 AM<br> Shift length: 8 - 10 hours<br> 5 days/week, must be available any day </p> <br><b>Benefits</b><ul><li>Health, dental, vision insurance - available after 90 days</li> <li>Paid time off</li> <li>401(k) plan</li> </ul> <br><b>Qualifications</b><ul><li>Must pass drug screen</li> <li>Can lift 50 lbs</li> <li>Must be at least 18+ years old</li> </ul> Bachelor's Degree or high school diploma/GED with at least 4 years experience in customer service<br>Ability to read, analyze and interpret general business periodicals, professional journals, and technical procedures.<br>Ability to effectively present information and respond to questions from managers, clients, and general public<br>Ability to calculate figures, and amounts such as discounts, interest, proportions, percentages, area, mass and volume.<br>Effective oral and written communication skills.<br>High level of interpersonal skills to handle sensitive and confidential situation and documentation.<br>Computer Literacy <br><br> <b>About Restaurant Depot</b><br> <p>Restaurant Depot is a Members-Only Wholesale Cash & Carry Foodservice Supplier. Their mission is to be your one-stop shop for savings, selection, and service, seven days a week. They have been supplying independent food businesses with quality products from large cash and carry warehouse stores since 1990. They became the leading low-cost alternative to other foodservice suppliers by eliminating the overhead of a traditional distributor, focusing on the needs of independent foodservice operators and offering free membership.</p>"
""",
"""Analyze this job posting and create a unique rewritten version in the same language. Return as minified JSON with format {"titles": NEW_TITLE, "descriptions": NEW_DESCRIPTION, "lang_code": LANGUAGE_CODE}. Make it engaging while keeping core requirements: "Buscamos un/a administrativo/a para trabajar en empresa del sector de la alimentación situada en la comarca de la Selva. Las tareas que tendrá que realizar son las siguientes:- Investigación y análisis de accidentes laborales.<br>- Participación en auditoría interna.<br>- Actualización de la normativa vigente.<br>- Gestión de reconocimientos médicos.<br>- Otras tareas relacionadas con el puesto de trabajo- Experiencia mínima de 1 año.<br>- Buscamos a una persona dinámica, proactiva, resolutiva y con ganas de trabajar.<br>- Valorable residencia próxima al puesto de trabajo"
"""
]

# temperature=1.0
# top_p=1.0
#top_k=50,

def main(args: argparse.Namespace):
    client = OpenAI(
        base_url=args.endpoint,
        api_key=api_key,
    )
    stream = False
    enable_json_mode = args.json
    max_tokens = 512
    extra_body = None
    response_format = None

    if enable_json_mode:
        extra_body={
            "guided_json": {"type": "object"},
        }

    for i in range(1):
        for j, user_prompt in enumerate(user_prompts):
            print(f"[enable json mode: {enable_json_mode}] Run Case {i} - {j}")
            start_time = time.time()
            chat_completion = client.chat.completions.create(
                temperature=0.9,
                top_p=0.9,
                stream=stream,
                model=f"{args.model}",
                messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }],
                max_tokens=max_tokens,
                extra_body=extra_body,
                response_format=response_format,
            )
            if stream:
                for chunk in chat_completion:
                    print(chunk.choices[0].delta.content or "", end="")
            else:
                #print(chat_completion.usage)
                print(chat_completion.choices[0].message.content)
            end_time = time.time()
            print(f"\nTime taken: {end_time - start_time:.2f} seconds, {chat_completion.usage.completion_tokens} tokens")
            print("----------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON generation evaluation"
    )
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--model", type=str, help="The model name")
    parser.add_argument("--json", action="store_true", help="Enable JSON output if set")

    args = parser.parse_args()
    main(args)


