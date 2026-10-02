import pandas as pd


def generate_recommendations(
    customer,
    explanation,
    domain
):

    recommendations = []


    # ============================================================
    # TELECOM
    # ============================================================

    if domain == "telecom":

        if "contract" in customer.index:
            contract = str(customer["contract"]).lower()

            if "month" in contract:
                recommendations.append(
                    "Consider offering a suitable longer-term contract option or renewal incentive."
                )

        if "onlinesecurity" in customer.index:

            security = str(
                customer["onlinesecurity"]
            ).lower()

            if security == "no":
                recommendations.append(
                    "Consider offering an appropriate online security service or security-focused plan."
                )

        if "techsupport" in customer.index:

            support = str(
                customer["techsupport"]
            ).lower()

            if support == "no":
                recommendations.append(
                    "Offer technical support assistance or a support-enabled service option."
                )

        if "paymentmethod" in customer.index:

            recommendations.append(
                "Review the customer's payment experience and provide convenient alternative payment options."
            )

        if "monthly_charges" in customer.index:

            recommendations.append(
                "Review the customer's current internet plan and provide suitable plan or service options."
            )


    # ============================================================
    # BANKING
    # ============================================================

    elif domain == "banking":

        if "active_member" in customer.index:

            try:

                if float(
                    customer["active_member"]
                ) == 0:

                    recommendations.append(
                        "Consider a targeted engagement program to encourage active use of banking services."
                    )

            except:
                pass


        if "products_number" in customer.index:

            try:

                if float(
                    customer["products_number"]
                ) <= 1:

                    recommendations.append(
                        "Consider offering relevant banking products based on the customer's needs."
                    )

            except:
                pass


        if "tenure" in customer.index:

            try:

                if float(
                    customer["tenure"]
                ) <= 2:

                    recommendations.append(
                        "Provide an early-stage customer engagement program to strengthen retention."
                    )

            except:
                pass


        if "credit_score" in customer.index:

            try:

                if float(
                    customer["credit_score"]
                ) < 600:

                    recommendations.append(
                        "Consider providing suitable financial guidance or relevant banking support."
                    )

            except:
                pass


    # ============================================================
    # EMPLOYEE
    # ============================================================

    elif domain == "employee":

        if "overtime" in customer.index:

            overtime = str(
                customer["overtime"]
            ).lower()

            if overtime == "yes":

                recommendations.append(
                    "Review workload and overtime patterns and consider workload balancing measures."
                )


        if "jobsatisfaction" in customer.index:

            try:

                if float(
                    customer["jobsatisfaction"]
                ) <= 2:

                    recommendations.append(
                        "Consider employee engagement and job satisfaction initiatives."
                    )

            except:
                pass


        if "worklifebalance" in customer.index:

            try:

                if float(
                    customer["worklifebalance"]
                ) <= 2:

                    recommendations.append(
                        "Consider work-life balance initiatives and flexible working arrangements where appropriate."
                    )

            except:
                pass


        if "yearsincurrentrole" in customer.index:

            try:

                if float(
                    customer["yearsincurrentrole"]
                ) >= 5:

                    recommendations.append(
                        "Consider career progression and role-development opportunities."
                    )

            except:
                pass


    # ============================================================
    # E-COMMERCE
    # ============================================================

    elif domain == "ecommerce":

        # --------------------------------------------------------
        # COMPLAINT
        # --------------------------------------------------------

        if "complain" in customer.index:

            try:

                if float(
                    customer["complain"]
                ) == 1:

                    recommendations.append(
                        "Prioritize complaint resolution and follow up with the customer after the issue is addressed."
                    )

            except:
                pass


        # --------------------------------------------------------
        # SATISFACTION SCORE
        # --------------------------------------------------------

        if "satisfactionscore" in customer.index:

            try:

                satisfaction = float(
                    customer["satisfactionscore"]
                )

                if satisfaction <= 2:

                    recommendations.append(
                        "Consider a targeted customer satisfaction improvement program based on the customer's recent experience."
                    )

                elif satisfaction == 3:

                    recommendations.append(
                        "Consider proactive engagement to improve the customer's overall shopping experience."
                    )

            except:
                pass


        # --------------------------------------------------------
        # DAYS SINCE LAST ORDER
        # --------------------------------------------------------

        if "daysincelastorder" in customer.index:

            try:

                days = float(
                    customer["daysincelastorder"]
                )

                if days >= 15:

                    recommendations.append(
                        "Consider a re-engagement campaign with relevant offers to encourage the customer to place another order."
                    )

            except:
                pass


        # --------------------------------------------------------
        # ORDER COUNT
        # --------------------------------------------------------

        if "ordercount" in customer.index:

            try:

                orders = float(
                    customer["ordercount"]
                )

                if orders <= 2:

                    recommendations.append(
                        "Consider personalized offers or onboarding incentives to encourage repeat purchases."
                    )

            except:
                pass


        # --------------------------------------------------------
        # CASHBACK
        # --------------------------------------------------------

        if "cashbackamount" in customer.index:

            try:

                cashback = float(
                    customer["cashbackamount"]
                )

                if cashback < 100:

                    recommendations.append(
                        "Consider relevant cashback or loyalty incentives to encourage continued engagement."
                    )

            except:
                pass


        # --------------------------------------------------------
        # WAREHOUSE TO HOME
        # --------------------------------------------------------

        if "warehousetohome" in customer.index:

            try:

                distance = float(
                    customer["warehousetohome"]
                )

                if distance >= 15:

                    recommendations.append(
                        "Review delivery experience and provide suitable delivery or logistics support for customers with longer delivery distances."
                    )

            except:
                pass


        # --------------------------------------------------------
        # APP ENGAGEMENT
        # --------------------------------------------------------

        if "hourspendonapp" in customer.index:

            try:

                hours = float(
                    customer["hourspendonapp"]
                )

                if hours < 2:

                    recommendations.append(
                        "Consider personalized app engagement campaigns to encourage regular interaction with the platform."
                    )

            except:
                pass


    # ============================================================
    # OTT
    # ============================================================

    elif domain == "ott":

        if "watch_hours" in customer.index:

            try:

                if float(
                    customer["watch_hours"]
                ) < 5:

                    recommendations.append(
                        "Consider personalized content recommendations and engagement campaigns."
                    )

            except:
                pass


        if "subscription_type" in customer.index:

            recommendations.append(
                "Consider offering a suitable subscription option based on the customer's usage pattern."
            )


    # ============================================================
    # FALLBACK
    # ============================================================

    if not recommendations:

        recommendations.append(
            "Continue monitoring the customer's behavior and consider proactive retention engagement."
        )


    return recommendations