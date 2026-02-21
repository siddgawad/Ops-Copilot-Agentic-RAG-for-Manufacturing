from pydantic import BaseModel, Field 

#We will use object-oriented programming - first we create a class that arbitrarily defines what a Query is 

class QueryRequest(BaseModel):
    #this must be a string and we provide metadata for the swagger ui 
    question: str = Field(..., 
    description="The operations question from the user")

    #this is an optional string defaulting to None 
    machine_id: str | None = Field(default=None, description="Optional ID of machine involved")

class QueryResponse(BaseModel):
    #this must eb a string containing the A's answer 
    answer: str

    