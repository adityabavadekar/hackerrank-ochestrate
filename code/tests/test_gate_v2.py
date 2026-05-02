from code import gate
from code.config import SAMPLE_SUPPORT_TICKETS_CSV
from code.models import Company
from code.models import SupportTicket
from code.gate import _is_low_quality
from code.config import DEFAULT_INPUT_CSV

from code.pipeline import _load_tickets

tickets = _load_tickets(DEFAULT_INPUT_CSV)
sample_tickets = _load_tickets(SAMPLE_SUPPORT_TICKETS_CSV)

tickets.extend(sample_tickets)

tickets.append(SupportTicket(issue="aaa", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="----------------------", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="------------", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="asdfhakjsdhfakhfdkajhfdkj", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="akjhakvhvkahkd", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="qwertyuiop", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="zxcvbnm", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="brghtqzwnp", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="hshdfbcnmv", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="aaaaaaaaaaaaa", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="pls pls pls pls pls pls pls pls pls", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="help help help help help help help", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="1234567890123456", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="9999999999999", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="!@#$% ^&*() _+", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="??? !!! ???", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="a", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="no", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="ok", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="  \n \t  ", subject="", company=Company.HACKERRANK)) # Whitespace only
tickets.append(SupportTicket(issue="API down", subject="", company=Company.CLAUDE))
tickets.append(SupportTicket(issue="site broken", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="card stolen", subject="", company=Company.VISA))
tickets.append(SupportTicket(issue="My Visa card ending in 1234567890123456 was declined", subject="", company=Company.VISA))
tickets.append(SupportTicket(issue="Trace ID: asdfghjkl1234567890qwertyuiop Please investigate this failure.", subject="", company=Company.CLAUDE))
tickets.append(SupportTicket(issue="Why is my regex ^[a-zA-Z0-9]+$ failing on the compiler?", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="Getting a `[object Object]` error when clicking submit.", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="Cannot access https://support.hackerrank.com/articles/4811403281", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="Experiencing intermittent authentication vulnerabilities", subject="", company=Company.HACKERRANK))
tickets.append(SupportTicket(issue="Troubleshooting my asynchronous concurrency architecture", subject="", company=Company.CLAUDE))
tickets.append(SupportTicket(issue="My transaction failed. Reference: b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee", subject="", company=Company.CLAUDE))

for i, ticket in enumerate(tickets):
    result = gate.run(ticket)
    print(f"[PASSED: {result.passed}] [HAS_RESULT: {result.early_result is not None}] {ticket.full_text()}")